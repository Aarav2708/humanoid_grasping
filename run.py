# import subprocess

# # Run with shell=True if using environment variables or sourcing files
# subprocess.run("/home/hpm-mv/parent_graspnet/humanoid_grasping/run_all.sh", shell=True)
import subprocess

subprocess.run([
    "gnome-terminal",
    "--",
    "bash",
    "-c",
    "/home/hpm-mv/parent_graspnet/humanoid_grasping/run_all.sh; exec bash"
])
