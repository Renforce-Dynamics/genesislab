from ..utils.importer import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place", "direct.humanoid_amp.motions"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)