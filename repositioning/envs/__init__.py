"""Helper function for creating environments."""

from .repositioning_env import RepositioningEnv
from .panda_human_env import PandaHumanRepositioningEnv
from .two_link_env import TwoLinkRepositioningEnv


def create_env(name: str) -> RepositioningEnv:
    """Helper function for creating environments."""

    if name == "panda-human":
        return PandaHumanRepositioningEnv()
    
    if name == "two-link":
        return TwoLinkRepositioningEnv()
    
    raise NotImplementedError
