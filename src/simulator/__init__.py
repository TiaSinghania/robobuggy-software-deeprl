from gymnasium.envs.registration import register

register(
    id="BuggyCourseEnv-v1",
    entry_point="src.simulator.environment:BuggyCourseEnv",
)
