from .environment import CatchingEnvExpert
from .scenarios import load


def MPECatchingEnvExpert(args):
    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = CatchingEnvExpert(world, reset_callback=scenario.reset_world, reward_callback=scenario.reward, 
                        observation_callback= scenario.observation, info_callback=  scenario.info, 
                        done_callback=scenario.if_done, post_step_callback=scenario.post_step)

    return env
