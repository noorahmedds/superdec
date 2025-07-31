import numpy as np
import os
import viser
import time
from superdec.utils.predictions_handler import PredictionHandler 
from superdec.utils.visualizations import generate_ncolors
import torch
import trimesh

RESOLUTION = 30

def main():
  server = viser.ViserServer()
  
  epoch = 1000
  dataset = 'scene'
  split = '95748dd597'
  name_experiment = '/ephemeral/outputs/28-07-normalized-second'
  base_path = '/ephemeral/outputs'
  gt = False
  gt_suffix = '_gt' if gt else ''
  
  input_path = os.path.join(base_path, name_experiment, f'{dataset}_{str(epoch)}_{split}_rdm.npz')

  print("Opening npz...")
  predictions_sq = PredictionHandler.from_npz(input_path)
  print("Computing meshes...")
  meshes = predictions_sq.get_meshes(resolution=RESOLUTION)

  existence_mesh = torch.ones(len(meshes), dtype = torch.bool)
  print("Computing point clouds...")
  pcs = predictions_sq.get_segmented_pcs()
  print("Done!")
  names = predictions_sq.names

  if dataset == 'scene': # visualization for scenes
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    int_ids = np.array([int(name) for name in names])
    #colors = generate_ncolors(int_ids.max()+1)/255.0 # Color each object with a different color
    colors = generate_ncolors(max(int_ids)+1)/255 #(max(int_ids)+1)/255

    # filter by iou
    # max_bounds = []
    # min_bounds = []
    # for mesh in meshes:
    #   if mesh is not None:
    #     max_bounds.append(mesh.bounds[1])
    #     min_bounds.append(mesh.bounds[0])
    # max_coord = np.max(np.array(max_bounds), axis=0)
    # min_coord = np.min(np.array(min_bounds), axis=0)


    # sampled_points = np.random.uniform(min_coord, max_coord, size=(10000, 3))
    # occ = predictions_sq.get_occupancy(sampled_points)
    # occ = torch.tensor((predictions_sq.exist > 0)) * occ
    # occ_max = occ.max(1)[0]
    
    # for i in range (len(meshes)):
    #   for k in range(occ.shape[1]):
    #     if predictions_sq.exist[i, k] < 0.5:
    #       continue
    #     for t in range(k + 1, occ.shape[1]):
    #       if predictions_sq.exist[i, t] < 0.5:
    #         continue
    #       intersection = torch.logical_and(occ[i, k],occ[i, t]).sum()
    #       union = torch.logical_or(occ[i, k],occ[i, t]).sum()
    #       iou = intersection / union if union > 0 else 0
    #       if iou != 0.0:
    #         print(f"Mesh {i}_{k} and {i}_{t} have iou {iou:.2f}")
    #       if iou > 0.2:
    #         print(f"Removing mesh {i}_{k} and {i}_{t} with iou {iou:.2f}")
    #         if predictions_sq.exist[i, k] < predictions_sq.exist[i, t]:
    #           predictions_sq.exist[i, k] = False
    #         else:
    #           predictions_sq.exist[i, t] = False
            

        
    #   for j in range (i+1, len(meshes)):
    #     intersection = torch.logical_and(occ_max[i],occ_max[j]).sum()
    #     union = torch.logical_or(occ_max[i],occ_max[j]).sum()
    #     iou = intersection / union if union > 0 else 0
    #     if iou != 0.0:
    #       print(f"Mesh {names[i]} and {names[j]} have iou {iou:.2f}")
    #     if iou > 0.3:
    #       print(f"Removing mesh {names[i]} and {names[j]} with iou {iou:.2f}")
    #       mesh_i_name = int(names[i])
    #       mesh_j_name = int(names[j])
    #       if mesh_i_name < mesh_j_name:
    #         existence_mesh[i] = False
    #       else:
    #         existence_mesh[j] = False
    # meshes = predictions_sq.get_meshes(resolution=RESOLUTION)


    for idx in range(len(meshes)):
      if meshes[idx] == None or not existence_mesh[idx] or int(names[idx]) == 0:# int(names[idx]) >= 50 ormeshes[idx] == None or int(names[idx]) >= 50: # TODO for now I am only taking the first 200 objects, modify this
        continue
      meshes[idx].visual.face_colors = np.ones((meshes[idx].visual.face_colors.shape[0], 3)) * colors[int(names[idx])]
      meshes[idx].visual.vertex_colors = np.ones((meshes[idx].visual.vertex_colors.shape[0], 3)) * colors[int(names[idx])]
      server.scene.add_mesh_trimesh(f"superquadrics_{names[idx]}", mesh=meshes[idx], visible=True)
    
      server.scene.add_point_cloud(
            name=f"/segmented_pointcloud_{names[idx]}",
            points=np.array(pcs[idx].points),
            colors=pcs[idx].colors,
            point_size=0.005,
            visible = False
        )
  else: # visualization for objects
    def draw_superquadric_and_segmentation():
      idx = int(gui_model_selection.value)
      server.scene.add_mesh_trimesh("superquadrics", mesh=meshes[idx], visible=True)
      
      server.scene.add_point_cloud(
            name="/segmented_pointcloud",
            points=np.array(pcs[idx].points),
            colors=np.array(pcs[idx].colors),
            point_size=0.005,
        )
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    gui_model_selection = server.gui.add_dropdown("Model index", [str(i) for i in range(len(names))], initial_value='0')
    gui_model_selection.on_update(lambda _: draw_superquadric_and_segmentation())
    draw_superquadric_and_segmentation()

  
    
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)
    
  
  while True:
      time.sleep(10.0)

if __name__ == '__main__':
  main()