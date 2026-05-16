# Architecture

The project is organized as layered modules: chess logic owns legal state, board mapping owns geometry calculations, move translation converts legal chess moves into physical actions, the registry tracks concrete MuJoCo body identities, the MuJoCo layer generates and loads XML, motion executes waypoint plans, health checks validate runtime state, and UI/debug modules provide interaction surfaces.

The chess layer does not import MuJoCo or motion modules. Motion and gripper modules do not import chess logic. The game loop is the orchestration boundary that validates moves, translates actions, executes physical work, updates registry state, and commits the logical board last.
