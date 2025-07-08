import numpy as np
import os
import viser
import time
from superdec.utils.predictions_handler import PredictionHandler 

RESOLUTION = 32

def main():
  server = viser.ViserServer()
  server.scene.set_up_direction([0.0, 1.0, 0.0])
  
  epoch = 42
  input_path = os.path.join('outputs', f'epoch_{str(epoch)}.npz')
  predictions_sq = PredictionHandler.from_npz(input_path)

  meshes = predictions_sq.get_meshes(resolution=RESOLUTION)
  pcs = predictions_sq.get_segmented_pcs()
  names = predictions_sq.names

  gui_model_selection = server.gui.add_dropdown("Model index", [str(i) for i in range(len(names))], initial_value='0')
  gui_model_selection.on_update(lambda _: draw_superquadric_and_segmentation())

  def draw_superquadric_and_segmentation():
    idx = int(gui_model_selection.value)
    server.scene.add_mesh_trimesh("superquadrics", mesh=meshes[idx], visible=True)
    
    server.scene.add_point_cloud(
          name="/segmented_pointcloud",
          points=np.array(pcs[idx].points),
          colors=np.array(pcs[idx].colors),
          point_size=0.005,
      )
    
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)
    
  draw_superquadric_and_segmentation()
  while True:
      time.sleep(10.0)

if __name__ == '__main__':
  main()