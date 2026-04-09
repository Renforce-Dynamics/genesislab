# Notes for developments

## TODO list

1. Sensor implementation not finished, where the ray caster and the camera is not well implemented
2. Motion tracking tasks, where the mdp is gathered but the task does not. This indicate that we need the motion reference for the tasks usage, together with a fk script to augment the observation.
3. Visual material, requires better define, where we could import a png file (Minecraft Style).
4. Debug vis for beyondmimic.
5. Usd Terrain importor or gaussian importer for better scene representation.
6. AMP surrport (Wait for RenforceRL to opensource).
7. SoccerLab surrport for Visualization.

Docs yet not prepared, take some time to write.

[Genesis] [23:17:02] [WARNING] omniverse-kit not found. USD baking will be disabled. Please install it with `pip install --extra-index-url https://pypi.nvidia.com omniverse-kit`. See https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/usd_import.html.

## Refactor

1. Clearify roles for engine and components

### Future works
1. Add locomotion benchmarks
2. Add vla task benchmarks

## Caution things

1. Effort clip for actuator. (currently using default value for bound. Not true)

# Done works

