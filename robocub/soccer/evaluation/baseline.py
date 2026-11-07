"""Typed scripted checkpoint: never interchangeable with a learned policy."""
def checkpoint_spec(args):
    if args.robot=='g1':
        motion='nominal_joint_hold' if args.backend=='isaaclab' else 'elbow_sine_0.05rad'
    else:motion='all_joint_sine_0.2rad'
    return {'kind':'field01_scripted_baseline','version':2,'backend':args.backend,
            'robot':args.robot,'field':args.field,'motion':motion,'learning':False}
