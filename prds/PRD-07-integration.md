# PRD-07: Integration

## Objective
Docker serving, ROS2 node, and ANIMA platform integration.

## Deliverables

### Docker
- `Dockerfile.serve`: 3-layer build from anima-serve:jazzy
- `docker-compose.serve.yml`: profiles serve, ros2, api, test
- `.env.serve`: module environment variables

### ROS2 Node
- `src/pilot/serve.py`: PiLoTNode(AnimaNode) subclass
  - `setup_inference()`: load weights, init feature net + JNGO
  - `process(image_msg)`: extract features, run JNGO, return PoseStamped
  - Subscribes: /camera/image_raw (sensor_msgs/Image)
  - Publishes: /pilot/pose (geometry_msgs/PoseStamped)
  - Publishes: /pilot/target_location (geometry_msgs/PointStamped)

### API Endpoints
- POST /predict: query image -> 6-DoF pose
- GET /health, /ready, /info: standard ANIMA endpoints

### HuggingFace
- Push checkpoint: `ilessio-aiflowlab/project_pilot-checkpoint`
- Model card with metrics, architecture diagram, usage

## Acceptance Criteria
- `docker compose --profile api up` starts API server
- /health returns 200 with module info
- /predict accepts image, returns pose JSON
- ROS2 node publishes PoseStamped on /pilot/pose
