import os
from pathlib import Path
from dataclasses import dataclass
import json
import glob
import h5py
import sparse
from pint import UnitRegistry
import numpy as np
import logging
from typing import List


@dataclass
class ExtractionData:
    plane_name: str = None
    dims: List[int] = None
    rois: np.array = None
    um_per_px: float = None
    

def get_rois_from_suite2p(stat_file, ops_file):
    stat = np.load(stat_file, allow_pickle=True)
    ops = np.load(ops_file, allow_pickle=True)
    ops = ops.item()
    dims = ops['Ly'], ops['Lx']    
    rois = np.empty((len(stat),) + dims, dtype=np.float32)
    
    for i, s in enumerate(stat):
        rois[i, s["ypix"], s["xpix"]] = s["lam"]

    return rois

def get_rois_from_extraction_h5(extraction_file):
    with h5py.File(extraction_file) as f:
        rois = sparse.COO(
            f["rois/coords"], f["rois/data"], f["rois/shape"]
        ).todense()
        return rois



def load_session_planes(session_dir, default_um_per_px=0.78):    
    session_dir = Path(session_dir)
    planes = []
    
    for plane_name in sorted(os.listdir(session_dir)):
        plane_dir = session_dir / plane_name
        if plane_dir.is_dir() and plane_dir.parts[-1] != 'nextflow': 
            
            extraction = ExtractionData(plane_name=plane_name)            
            
            try:
                session_file = next(session_dir.glob("session.json"))            
                extraction.um_per_px, extraction.dims = get_plane_metadata(session_file)
            except StopIteration as e:
                extraction.um_per_px = 0.78

            try:
                extraction_file = next(plane_dir.glob("**/extraction.h5"))            
                extraction.rois = get_rois_from_extraction_h5(extraction_file)    
            except StopIteration as e:
                try:
                    stat_file = next(plane_dir.glob("**/stat.npy"))                
                    ops_file = next(plane_dir.glob("**/ops.npy"))
                    extraction.rois = get_rois_from_suite2p(stat_file, ops_file)                    
                except StopIteration as e:                
                    logging.error(f"could not find rois for {plane_dir}")
                    continue

            
            
            planes.append(extraction)
                
    return planes

def load_bci_session_planes(session_dir):
    metadata_file = f"{session_dir}/metadata.json"
    
    extraction = ExtractionData(plane_name=session_dir)
    
    with open(metadata_file, 'r') as f:
        j = json.load(f)
        extraction.um_per_px=j['fov_scale_factor']
        
    stat_path=f"{session_dir}/suite2p-BCI/plane0/stat.npy"
    ops_path=f"{session_dir}/suite2p-BCI/plane0/ops.npy"
              
    assert os.path.exists(stat_path), stat_path
    assert os.path.exists(ops_path), ops_path
    
    extraction.rois = get_rois_from_suite2p(stat_path, ops_path)
    
    return extraction

def get_plane_metadata(session_file):
    """ get um_per_pixel and dims (FOV size) from session.json """
    
    with open(session_file, "r") as j:
        session_data = json.load(j)

    for data_stream in session_data["data_streams"]:
        if data_stream.get("ophys_fovs"):
            fov = data_stream["ophys_fovs"][0]
            um_per_pixel = (
                float(fov["fov_scale_factor"])
                * (UnitRegistry().parse_expression(fov["fov_scale_factor_unit"]))
                .to("um/pixel")
                .magnitude
            )
            dims = (fov["fov_height"], fov["fov_width"])
    return um_per_pixel, dims
