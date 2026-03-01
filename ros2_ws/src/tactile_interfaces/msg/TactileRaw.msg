std_msgs/Header header

uint32 rows
uint32 cols
uint64 sequence_id
string frame_id

# Flattened tactile values of size rows * cols.
float32[] forces

# Optional per-taxel 3D force components of size rows * cols.
float32[] forces_fx
float32[] forces_fy
float32[] forces_fz
