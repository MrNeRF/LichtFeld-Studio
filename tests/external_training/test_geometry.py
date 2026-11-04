# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util,json,tempfile,types,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/python'))
from lfs_plugins.training_runtime import geometry as g

class GeometryContracts(unittest.TestCase):
 def test_blender_preview_and_cameras_share_studio_vertical_without_changing_dataset(self):
  value=self.value();D=np.asarray(g.FLIP);C=np.asarray(g.BLENDER_TO_VISUALIZER)
  point_source=np.array([2.,3.,11.,1.]);point_data=D@point_source
  np.testing.assert_allclose(g.visualizer_model_transform(value)@point_data,C@point_source)
  np.testing.assert_allclose(g.visualizer_model_transform(value)@D@np.array([0.,0.,1.,0.]),[0,1,0,0])
  camera=g.matrix(value['cameras'][0]['camera_to_data'])
  P=g.viewer_data_transform(value)
  np.testing.assert_allclose(np.linalg.inv(P@camera)@g.viewer_model_transform(value)@point_data,
                             np.linalg.inv(camera)@point_data)
  np.testing.assert_allclose(g.source_model_transform(value)@point_data,point_source)
 def test_native_sessions_keep_their_existing_viewer_frame(self):
  value={'source_frame':'lichtfeld-visualizer','model_to_data':np.eye(4).tolist()}
  np.testing.assert_allclose(g.viewer_model_transform(value),np.eye(4))
  np.testing.assert_allclose(g.visualizer_model_transform(value),g.FLIP)
 def test_rotated_camera_preserves_pixels_across_all_frames(self):
  # Blender cameras look along -Z with +Y up; dataset cameras use +Z / -Y.
  angle=.63;c=np.cos(angle);s=np.sin(angle)
  camera_source=np.array([[c,-s,0,4],[s,c,0,-3],[0,0,1,7],[0,0,0,1.]])
  frame={'file_path':'images/rotated.png','transform_matrix':camera_source.tolist(),
         'w':1280,'h':720,'fl_x':910,'fl_y':920,'cx':641,'cy':359}
  value=g.from_capture([frame]);D=np.asarray(g.FLIP)
  point_source=camera_source@np.array([.7,-.2,-4,1])
  point_data=D@point_source
  camera_data=g.matrix(value['cameras'][0]['camera_to_data'])
  point_camera=np.linalg.inv(camera_data)@point_data
  np.testing.assert_allclose(point_camera,[.7,.2,4,1])
  def project(point):return [910*point[0]/point[2]+641,920*point[1]/point[2]+359]
  P=g.viewer_data_transform(value)
  point_view=np.linalg.inv(D@P@camera_data)@(D@P@point_data)
  np.testing.assert_allclose(project(point_view),project(point_camera))
  np.testing.assert_allclose(g.source_model_transform(value)@point_data,point_source)
 def value(self):
  pose=np.eye(4);pose[:3,3]=[3,5,7]
  return g.from_capture([{'file_path':'images/view.png','transform_matrix':pose.tolist(),'w':640,'h':480,'fl_x':500,'fl_y':510,'cx':310,'cy':235}])
 def test_point_and_camera_round_trip_is_one_world_conversion(self):
  value=self.value();point=np.array([2.,3.,11.,1.]);D=np.array(g.FLIP)
  np.testing.assert_allclose(g.source_model_transform(value)@D@point,point)
  camera=g.matrix(value['cameras'][0]['camera_to_data'])
  np.testing.assert_allclose(D@camera@D,np.array([[1,0,0,3],[0,1,0,5],[0,0,1,7],[0,0,0,1]]))
 def test_model_node_transform_is_preserved(self):
  value=self.value();M=np.eye(4);M[:3,3]=[13,17,19];value['model_to_data']=M.tolist()
  np.testing.assert_allclose(g.source_model_transform(value),np.array(g.FLIP)@M)
 def test_reject_invalid_and_unbounded_metadata(self):
  for key,bad in [('schema',True),('data_to_source',np.zeros((4,4)).tolist()),('cameras',[{}])]:
   value=self.value();value[key]=bad
   with self.assertRaises((ValueError,TypeError)):g.validate(value)
  with self.assertRaises(ValueError):g.load('.',{'path':'../view_geometry.json','bytes':1})
 def test_legacy_model_frame_is_explicit_default(self):
  np.testing.assert_array_equal(g.source_model_transform(),g.FLIP)
 def test_immutable_reference_checks_size_hash_and_path(self):
  with tempfile.TemporaryDirectory() as root:
   state={}
   publisher=types.SimpleNamespace(root=Path(root),update=lambda **values:state.update(values))
   g.publish(publisher,self.value())
   loaded=g.load(root,state['geometry'])
   self.assertEqual(loaded['cameras'][0]['name'],'view.png')
   (Path(root)/'view_geometry.json').write_text('{}')
   with self.assertRaises(ValueError):g.load(root,state['geometry'])

if __name__=='__main__':unittest.main()
