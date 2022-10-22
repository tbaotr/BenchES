from .control_gym import Master


def get_problem(type, params):
    if type == "gym":
        return Master(params)


__all__ = [
    "get_problem",
]
