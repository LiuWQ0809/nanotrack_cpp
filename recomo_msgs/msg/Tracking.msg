# Tracking message for vision tracker output
# Published on /recomo/tracking topic

std_msgs/Header header

# Unique tracking ID (e.g., "track_0001")
string tracking_id

# Tracking state: "pending", "tracking", or "lost"
string state

# Bounding box of tracked object (pixel coordinates)
sensor_msgs/RegionOfInterest bbox

# Tracking confidence (0.0 to 1.0)
float32 confidence
