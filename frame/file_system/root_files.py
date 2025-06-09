from pathlib import Path
from typing import Optional
import ROOT  # type: ignore


def explore_root_file(file_name: Path) -> None:
    """Explore the contents of a ROOT file"""
    print(f"\n=== Exploring ROOT file: {file_name} ===")
    
    with ROOT.TFile.Open(str(file_name), "READ") as root_file:
        if not root_file or root_file.IsZombie():
            print(f"Error: Cannot open file {file_name}")
            return
        
        print(f"File size: {root_file.GetSize()} bytes")
        print(f"File format version: {root_file.GetVersion()}")
        
        # List all keys (objects) in the file
        print("\n--- Objects in file ---")
        keys = root_file.GetListOfKeys()
        for i, key in enumerate(keys):
            obj_name = key.GetName()
            obj_class = key.GetClassName()
            obj_title = key.GetTitle()
            print(f"{i+1}. {obj_name} (Type: {obj_class}, Title: {obj_title})")
            
            # Get the object and examine it
            obj = key.ReadObj()
            if obj:
                _examine_object(obj, obj_name, obj_class)


def _examine_object(obj, name: str, class_name: str) -> None:
    """Examine a specific ROOT object"""
    print(f"   Details for {name}:")
    
    if "TH1" in class_name or "TH2" in class_name:
        # It's a histogram
        print(f"     Entries: {obj.GetEntries()}")
        print(f"     Mean: {obj.GetMean():.3f}")
        print(f"     RMS: {obj.GetRMS():.3f}")
        print(f"     Bins: {obj.GetNbinsX()}")
        if "TH2" in class_name:
            print(f"     Y-Bins: {obj.GetNbinsY()}")
        print(f"     X-axis: [{obj.GetXaxis().GetXmin():.2f}, {obj.GetXaxis().GetXmax():.2f}]")
        
    elif "TTree" in class_name:
        # It's a tree
        print(f"     Entries: {obj.GetEntries()}")
        print(f"     Branches:")
        branches = obj.GetListOfBranches()
        for i, branch in enumerate(branches):
            if i < 10:  # Show first 10 branches
                print(f"       - {branch.GetName()} ({branch.GetTitle()})")
            elif i == 10:
                print(f"       ... and {branches.GetSize() - 10} more branches")
                break
                
    elif "TGraph" in class_name:
        # It's a graph
        print(f"     Points: {obj.GetN()}")
        
    elif "TF1" in class_name:
        # It's a function
        print(f"     Formula: {obj.GetExpFormula()}")
        print(f"     Range: [{obj.GetXmin():.2f}, {obj.GetXmax():.2f}]")


def load_root(file_name: Path, object_name: Optional[str] = None):
    """Load object(s) from ROOT file. If object_name is None, returns all objects."""
    with ROOT.TFile.Open(str(file_name), "READ") as root_file:
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(f"Cannot open ROOT file: {file_name}")
        
        if object_name:
            # Load specific object
            obj = root_file.Get(object_name)
            if not obj:
                raise KeyError(f"Object '{object_name}' not found in {file_name}")
            
            # Clone the object so it persists after file is closed
            if hasattr(obj, 'Clone'):
                obj_clone = obj.Clone()
                if hasattr(obj_clone, 'SetDirectory'):
                    obj_clone.SetDirectory(0)
                return obj_clone
            return obj
        else:
            # Load all objects
            objects = {}
            keys = root_file.GetListOfKeys()
            for key in keys:
                obj = key.ReadObj()
                if obj and hasattr(obj, 'Clone'):
                    obj_clone = obj.Clone()
                    if hasattr(obj_clone, 'SetDirectory'):
                        obj_clone.SetDirectory(0)
                    objects[key.GetName()] = obj_clone
                elif obj:
                    objects[key.GetName()] = obj
            return objects

