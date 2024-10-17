import torch
from pathlib import Path
import timm
from timm.models.vision_transformer import _convert_dinov2


class TUMViTG(torch.nn.Module):
    patch_size = 14
    feature_dim = 1536

    def __init__(self, pretrained_path: Path, output_mode: str = "class"):
        """
        Args:
            pretrained_path (Path): Path to the pretrained model weights.
            output_mode (str): The output mode of the model. Choose from 'class' or 'class+mean'.
                'class' returns the class token, 'class+mean' returns the class token concatenated with the mean of the patch tokens.
        """
        super().__init__()

        if output_mode == "class+mean":
            self.feature_dim = TUMViTG.feature_dim * 2
        elif output_mode == "class":
            self.feature_dim = TUMViTG.feature_dim
        else:
            raise ValueError(
                f"Invalid output_mode: {output_mode}. Choose 'class' or 'class+mean'."
            )
        self.output_mode = output_mode

        self.output_mode = output_mode

        self.model = self.prep_model(pretrained_path)

    @staticmethod
    def prep_model(pretrained_path: Path) -> torch.nn.Module:
        """
        Returns model with the pretrained weights loaded.
        """
        model: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitg14"
        )
        # Load finetuned weights
        pretrained_model = torch.load(
            pretrained_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        # Reduce position embedding size to be able to load the pretrained weights
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 1536))
        model.load_state_dict(pretrained_model, strict=True)

        return model

    def freeze(self):
        """
        Freezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor):
        """
        Returns the original output of the model.
        See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L254
        for the original implementation.

        Returns:
            dict: { "x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm", "masks" }
        """
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Actual output depends on the self.output_mode set during initialization.

        Returns:
            torch.Tensor: `(batch_size, 1536)` or `(batch_size, 3072)` based on self.output_mode.
        """
        output = self.forward_features(x)

        if self.output_mode == "class":
            return output["x_norm_clstoken"]
        else:
            class_token = output["x_norm_clstoken"]
            patch_tokens = output["x_norm_patchtokens"]
            patch_mean = patch_tokens.mean(dim=1)
            return torch.cat([class_token, patch_mean], dim=-1)


def resample_abs_pos_embed(
    posemb: torch.Tensor,
    new_size: tuple[int, int],
    old_size: tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    """
    Taken from pytorch-image-models (timm) by Ross Wightman. Source:
    https://github.com/huggingface/pytorch-image-models/blob/e3242a52584bbc69f848f762d254e8a23932832c/timm/layers/pos_embed.py#L17
    """
    num_prefix_tokens = 1
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    posemb_prefix, posemb = (
        posemb[:, :num_prefix_tokens],
        posemb[:, num_prefix_tokens:],
    )

    # Do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # Interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = torch.nn.functional.interpolate(
        posemb,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # Add back extra (class, etc.) prefix tokens
    posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        print(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb


class TUMViTG_modified(torch.nn.Module):
    """
    Modified version of the TUMViTG model from the original implementation.

    This implementation does the position resampling beforehand and therefore does not support arbitrary input resolutions like the original implementation,
    but it avoids the overhead of position embedding interpolation in the forward pass and allows rectangular input resolutions.

    Meaning: use this class when you want to fine-tune the model for a specific resolution other than 224x224.
    """

    patch_size = 14
    feature_dim = 1536

    def __init__(
        self,
        pretrained_path: Path,
        output_mode: str = "class",
        img_size: tuple[int, int] = (224, 224),
    ):
        """
        Args:
            pretrained_path (Path): Path to the pretrained model weights.
            output_mode (str): The output mode of the model.
                'class' returns the class token, 'class+mean' returns the class token concatenated with the mean of the patch tokens.
            img_size (tuple[int, int]): The size of the input image. Defaults to (224, 224).
        """
        super().__init__()

        if output_mode == "class+mean":
            self.feature_dim = TUMViTG_modified.feature_dim * 2
        elif output_mode == "class":
            self.feature_dim = TUMViTG_modified.feature_dim
        else:
            raise ValueError(
                f"Invalid output_mode: {output_mode}. Choose 'class' or 'class+mean'."
            )
        self.output_mode = output_mode

        # check that the img_size is divisible by the patch_size
        if img_size[0] % self.patch_size != 0 or img_size[1] % self.patch_size != 0:
            raise ValueError(
                f"Image size {img_size} must be divisible by the patch size {self.patch_size} in both dimensions."
            )
        self.img_size = img_size

        self.model = self.prep_model(pretrained_path, img_size)

    @classmethod
    def prep_model(
        cls, pretrained_path: Path, img_size: tuple[int, int]
    ) -> torch.nn.Module:
        """
        Returns model with the pretrained weights loaded. Adapts the position embeddings to the new img_size.
        Args:
            pretrained_path (Path): Path to the pretrained model weights.
            img_size (tuple[int, int]): The size of the input image.
        """
        model: torch.nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitg14"
        )
        # Load finetuned weights
        pretrained_model = torch.load(
            pretrained_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        # Reduce position embedding size to be able to load the pretrained weights
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 1536))
        model.load_state_dict(pretrained_model, strict=True)

        if img_size != (224, 224):
            # Now, adjust the position embeddings for the new img_size using the resample_abs_pos_embed function
            old_size_x, old_size_y = 224 // cls.patch_size, 224 // cls.patch_size

            # Compute the new grid size
            grid_size_x = img_size[0] // cls.patch_size
            grid_size_y = img_size[1] // cls.patch_size

            # Resample the position embeddings
            model.pos_embed = torch.nn.Parameter(
                resample_abs_pos_embed(
                    posemb=model.pos_embed,
                    new_size=(grid_size_x, grid_size_y),
                    old_size=(old_size_x, old_size_y),
                    interpolation="bicubic",
                    antialias=True,
                    verbose=True,
                )
            )
        return model

    def freeze(self):
        """
        Freezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_attn(self):
        """
        Unfreeze the attention layers and position embeddings of the model.
        Useful for fine-tuning the model for larger resolutions.
        """
        for name, param in self.model.named_parameters():
            if ".attn." in name or "pos_embed" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> dict:
        """
        This implementation differs from the original implementation.
        It avoids calling `prepare_tokens_with_masks` (https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L213)
        as I couldn't make the rectangular position embeddings work with `self.interpolate_pos_encoding(x, w, h)` (https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179)

        See https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L254 for the original implementation.

        Returns:
            dict: { "x_norm_clstoken", "x_norm_patchtokens", "x_prenorm" }
        """
        x = self.model.patch_embed(x)
        x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.model.pos_embed
        for blk in self.model.blocks:
            x = blk(x)
        x_norm = self.model.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Actual output depends on the self.output_mode set during initialization.
        """
        output = self.forward_features(x)

        if self.output_mode == "class":
            return output["x_norm_clstoken"]
        else:
            class_token = output["x_norm_clstoken"]
            patch_tokens = output["x_norm_patchtokens"]
            patch_mean = patch_tokens.mean(dim=1)
            return torch.cat([class_token, patch_mean], dim=-1)


class timmTUMViTG(torch.nn.Module):
    patch_size = 14
    feature_dim = 1536

    def __init__(self, pretrained_path: Path, output_mode: str = "class", **timmkwargs):
        """
        Args:
            pretrained_path (Path): Path to the pretrained model weights.
            output_mode (str): The output mode of the model. Choose from 'class' or 'class+mean'.
                'class' returns the class token, 'class+mean' returns the class token concatenated with the mean of the patch tokens.
            **timmkwargs: Additional keyword arguments for the timm model.
        """
        super().__init__()

        if output_mode == "class+mean":
            self.feature_dim = TUMViTG.feature_dim * 2
        elif output_mode == "class":
            self.feature_dim = TUMViTG.feature_dim
        else:
            raise ValueError(
                f"Invalid output_mode: {output_mode}. Choose 'class' or 'class+mean'."
            )
        self.output_mode = output_mode

        self.output_mode = output_mode

        self.model = self.prep_model(pretrained_path, **timmkwargs)

    @staticmethod
    def prep_model(pretrained_path: Path, **timmkwargs) -> torch.nn.Module:
        model: torch.nn.Module = timm.create_model(
            "vit_giant_patch14_dinov2",
            pretrained=False,
            img_size=(224, 224),
            **timmkwargs,
        )
        pretrained = torch.load(
            pretrained_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        modified_pretrained = _convert_dinov2(pretrained, model)
        # Load finetuned weights
        model.load_state_dict(modified_pretrained)

        return model

    def freeze(self):
        """
        Freezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes all parameters of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def forward_features(self, x: torch.Tensor):
        """
        Returns the original output of the model.
        """
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Actual output depends on the self.output_mode set during initialization.

        Returns:
            torch.Tensor: `(batch_size, 1536)` or `(batch_size, 3072)` based on self.output_mode.
        """
        output = self.forward_features(x)

        if self.output_mode == "class":
            embedding = output[:, 0]
        else:
            class_token = output[:, 0]
            patch_tokens = output[:, 1:]
            patch_mean = patch_tokens.mean(dim=1)
            embedding = torch.cat([class_token, patch_mean], dim=-1)

        return embedding


def get_dinov2_TUMViTG(pretrained_path: Path):
    """
    Returns the original implementation of the TUMViTG model.
    """
    model: torch.nn.Module = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
    # Load finetuned weights
    pretrained_model = torch.load(
        pretrained_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    # Reduce position embedding size to be able to load the pretrained weights
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 1536))
    model.load_state_dict(pretrained_model, strict=True)
    return model


def get_timm_TUMViTG(pretrained_path: Path):
    """
    Returns the timm implementation of the TUMViTG model.
    """
    model: torch.nn.Module = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=False,
        img_size=(224, 224),
    )
    pretrained = torch.load(
        pretrained_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    modified_pretrained = _convert_dinov2(pretrained, model)
    # Load finetuned weights
    model.load_state_dict(modified_pretrained, strict=True)
    return model
