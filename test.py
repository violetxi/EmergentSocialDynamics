from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import cv2

# Initialize the environment
env = simple_spread_v3.parallel_env(N=3, render_mode='rgb_array')

# Reset the environment
env.reset()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('test.mp4', fourcc, 30, (700, 700))

for _ in range(100):
    # Choose random actions for each agent
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}    
    # Step the environment
    env.step(actions)
    
    # Render the environment
    frame = env.render()
    
    # Convert frame to BGR (for VideoWriter)
    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the video file
    video.write(frame)

# Close the video writer
video.release()

# Close the environment
env.close()
