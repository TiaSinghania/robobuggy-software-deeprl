from gym.envs.registration import register

register(
    id='BuggyCourseEnv-v1',
    entry_point='simulator.environment:BuggyCourseEnv',
    max_steps=300,
)
