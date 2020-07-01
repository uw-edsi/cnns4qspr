#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
This module contains functions for loading a pdb file and calculating
the atomic density fields for different atom types. The fields can then be
used for plotting, or to send into the convolutional neural network.
"""

import os
import sys
import torch
import numpy as np

from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2

import pickle

import numpy as np

import pybel
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
from silx.io.dictdump import dicttoh5
import h5py
import click


# In[4]:


device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type


# In[5]:


def load_pdb(path):

    pdb = PandasPdb().read_pdb(path)
# This just creates a dataframe from the pdb file using biopandas
#print('This is vars',vars(pdb))
    pdf = pdb.df['ATOM']
    x_coords = pdf['x_coord'].values
    y_coords = pdf['y_coord'].values
    z_coords = pdf['z_coord'].values
    atom_types = pdf['atom_name'].values
    residue_names = pdf['residue_name'].values
    
    pro_dict = generate_dict(x_coords, y_coords, z_coords, atom_types, residue_names)
    # add a value to the dictionary, which is all of the atomic coordinates just
    # shifted to the origin
    #protein_dict = shift_coords(protein_dict)

    return pro_dict



def load_mol2(path):
    
    mol = PandasMol2().read_mol2(path)
    pdf = mol
    x_coords = pdf.df['x'].values
    y_coords = pdf.df['y'].values
    z_coords = pdf.df['z'].values
    atom_types = pdf.df['atom_name'].values
    residue_names = pdf.df['subst_name'].values
    partial_charge = pdf.df['charge'].values
    smarts_notation = next(pybel.readfile('mol2', path))
    
    pro_dict = generate_dict(x_coords, y_coords, z_coords, atom_types, residue_names)

    pro_dict['charge'] = partial_charge
    pro_dict['smarts'] = smarts_notation

    # add a value to the dictionary, which is all of the atomic coordinates just
    # shifted to the origin
    #protein_dict = shift_coords(protein_dict)

    return pro_dict
    
def generate_dict(x, y, z, a_types, res_names):
    
    positions = []
    for i, xi in enumerate(x):
        position_tuple = (x[i], y[i], z[i])
        positions.append(position_tuple)
    positions = np.array(positions)

    # names of all the atoms contained in the protein
    
    num_atoms = len(a_types)
    atom_type_set = np.unique(a_types)
    num_atom_types = len(atom_type_set)

    # residue names
    
    residue_set = np.unique(res_names)

    pro_dict = {'x_coords':x, 'y_coords':y, 'z_coords':z,
                    'positions':positions, 'atom_types':a_types,
                    'num_atoms':num_atoms, 'atom_type_set':atom_type_set,
                    'num_atom_types':num_atom_types, 'residues':res_names,
                    'residue_set':residue_set}
    
    return pro_dict


def load_input(path, ligand=False):
    """
    Loads all of the atomic positioning/type arrays from a pdb/mol2 file.
    The arrays can then be transformed into density (or "field") tensors before
        being sent through the neural network.
    Parameters:
        path (str, required): The full path to the pdb file being voxelized.
    Returns:
        dictionary: A dictionary containing the following arrays from
            the pdb file: num_atoms, atom_types, positions, atom_type_set,
            xcoords, ycoords, zcoords, residues, residue_set
    """
    
    file_type = path.split('.')[-1]
    
    if file_type == 'pdb':
        protein_dict = load_pdb(path)
    
    elif file_type == 'mol2':
        protein_dict = load_mol2(path)
    else:
        raise ValueError('Need a pdb or mol2 file')
    # atomic coordinates
    if ligand == True:
        ligand_file = path.split('_')[:-1][0] + '_ligand.mol2'
        ligand_dict = load_mol2(ligand_file)
        mid_points = shift_coords(protein_dict, lig_dict=ligand_dict)
        #mid_points = shift_coords(ligand_dict, lig_dict=None)
        protein_dict['shifted_positions'] = protein_dict['positions'] - mid_points
        ligand_dict['shifted_positions'] = ligand_dict['positions'] - mid_points
        
        return protein_dict, ligand_dict
    else:
        mid_points = shift_coords(protein_dict)
        protein_dict['shifted_positions'] = protein_dict['positions'] - mid_points
        
        return protein_dict
        

    # create an array containing tuples of x,y,z for every atom
    
def get_extreme_values(name_dict):
    
    x_ext = np.array([name_dict['x_coords'].min(), name_dict['x_coords'].max()])
    y_ext = np.array([name_dict['y_coords'].min(), name_dict['y_coords'].max()])
    z_ext = np.array([name_dict['z_coords'].min(), name_dict['z_coords'].max()])
    
    return x_ext, y_ext, z_ext

def shift_coords(pro_dict, lig_dict=None):
    """
    This function shifts the coordinates of a protein so that it's coordinates are in the center
    of the field tensor.
    Parameters:
        protein_dict (dict): A dictionary of information from the first part of the load_input function.
    Returns:
        dictionary: The original protein dict but with an added value containing
            the coordinates of the protein shifted to the origin.
    """
    # find the extreme x, y, and z values that exist in the protein atomic coordinates
 
    x_extremes, y_extremes, z_extremes = get_extreme_values(pro_dict)
        

    if lig_dict:
        x_pro, y_pro, z_pro = get_extreme_values(pro_dict)
        x_lig, y_lig, z_lig = get_extreme_values(lig_dict)
        
        x_extremes = np.array([np.min(np.concatenate([x_pro, x_lig])), np.max(np.concatenate([x_pro, x_lig]))])
        y_extremes = np.array([np.min(np.concatenate([y_pro, y_lig])), np.max(np.concatenate([y_pro, y_lig]))])
        z_extremes = np.array([np.min(np.concatenate([z_pro, z_lig])), np.max(np.concatenate([z_pro, z_lig]))])
        
        #x_val = np.concatenate(x_pro, x_lig)
        #print(x_extremes)
    # calculate the midpoints of the extremes
    midpoints = [np.sum(x_extremes)/2, np.sum(y_extremes)/2, np.sum(z_extremes)/2]
#     print(x_extremes, y_extremes, z_extremes)

    # shift the coordinates by the midpoints of those extremes (center the protein on the origin)
#     protein_dict['shifted_positions'] = protein_dict['positions'] - midpoints

    return midpoints


# In[6]:


pro_dict = load_input('1a1e_pocket.pdb', ligand=True)


# In[7]:


path = '1a1e_ligand.mol2'
path.split('_')[:-1][0]


# In[8]:


def grid_positions(grid_array):
    """
    This function returns the 3D meshgrids of x, y and z positions.
    These cubes will be flattened, then used as a reference coordinate system
    to place the actual channel densities into.
    Parameters:
        grid_positions (pytorch tensor): lineraly spaced grid
    Returns:
        array: meshgrid array of the x, y and z positions.
    """
    xgrid = grid_array.view(-1, 1, 1).repeat(1, len(grid_array), len(grid_array))
    ygrid = grid_array.view(1, -1, 1).repeat(len(grid_array), 1, len(grid_array))
    zgrid = grid_array.view(1, 1, -1).repeat(len(grid_array), len(grid_array), 1)
    return (xgrid, ygrid, zgrid)

def norm_properties(prop, channel):
    
    if channel == 'hyb':
        max_prop = 3
        min_prop = 0
    elif channel == 'heterovalence':
        max_prop = 4
        min_prop = 0
    elif channel == 'heavyvalence':
        max_prop = 4
        min_prop = 0
    else:
        max_prop = 2.276
        min_prop = -1.167
        # clamping the charge value within a range
        # the range is chosen from ligand data
        prop[prop > 2.276] = 2.276
        prop[prop < -1.166] = -1.167
    
    norm_prop = (prop-min_prop)/(max_prop-min_prop)
    
    return norm_prop

def make_fields(protein_dict, channels, bin_size, num_bins, ligand_dict=None, ligand=False):
    """
    This function takes a protein dict (from load_input function) and outputs a
        large tensor containing many atomic "fields" for the protein.
    The fields describe the atomic "density" (an exponentially decaying function
        of number of atoms in a voxel) of any particular atom type.
    Parameters:
        protein_dict (dict, requred): dictionary from the load_input function
        channels (list-like, optional): the different atomic densities we want fields for
            theoretically these different fields provide different chemical information
            full list of available channels is in protein_dict['atom_type_set']
        bin_size (float, optional): the side-length (angstrom) of a given voxel in the box
            that atomic densities are placed in
        num_bins (int, optional): how big is the cubic field tensor side length
            (i.e., num_bins is box side length)
    Returns:
        dictionary: A list of atomic density tensors (50x50x50), one for each
            channel in channels
    """
    # sets of allowed filters to build channels with
    residue_filters = protein_dict['residue_set']
    atom_filters = protein_dict['atom_type_set']
    general_filters = ['all_C', 'all_O', 'all_N']
    residue_property_filters = np.array(['acidic', 'basic', 'polar', 'nonpolar',                                         'charged', 'amphipathic'])
    smart_filters = np.array(['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring'])
    named_prop = np.array(['hyb', 'heavyvalence', 'heterovalence', 'partialcharge'])
    protein_ligand_filters = np.array(['protein', 'ligand'])
    other_filters = np.array(['backbone', 'sidechains'])

    # consolidate into one set of filters
    filter_set = {'atom':atom_filters, 'residue':residue_filters,                  'residue_property':residue_property_filters,                   'smarts_property':smart_filters,                   'atom_property': named_prop, 'general': general_filters,
                  'protein_ligand': protein_ligand_filters, 'other':other_filters}

    # construct a single empty field, then initialize a dictionary with one
    # empty field for every channel we are going to calculate the density for
    empty_field = torch.zeros(num_bins, num_bins, num_bins).to(device_type)
    fields = {channel:empty_field for channel in channels}

    # create linearly spaced grid (default is -49 to 49 in steps of 2)
    grid_1d = torch.linspace(start=-num_bins / 2 * bin_size + bin_size / 2,
                             end=num_bins / 2 * bin_size - bin_size / 2,
                             steps=num_bins).to(device_type)

    # This makes three 3D meshgrids in for the x, y, and z positions
    # These cubes will be flattened, then used as a reference coordinate system
    # to place the actual channel densities into
    xgrid, ygrid, zgrid = grid_positions(grid_1d)

    for channel_index, channel in enumerate(channels):
        #print(channel)

        # no illegal channels allowed, assume the channel sucks
        channel_allowed = check_channel(channel, filter_set)

        if channel_allowed:
            pass
        else:
            #err_string = 'Allowed channels are: in a protein\'s atom_type_set,
                        # residue_set',or the \'sidechains\' and \'backbone\' channels.'
            raise ValueError('The channel ', channel, ' is not allowed for this protein.')


        # Extract positions of atoms that are part of the current channel

        atom_positions_protein = find_channel_atoms(channel, protein_dict, filter_set)
        
        
        if ligand == True:
            atom_positions_ligand = find_channel_atoms(channel, ligand_dict, filter_set)
            
            if channel == 'protein':
                    atom_positions = atom_positions_protein
        
            elif channel == 'ligand':
                    atom_positions = atom_positions_ligand
            else:
                atom_positions = np.concatenate((atom_positions_protein, atom_positions_ligand))
            #print(atom_positions_protein.shape)
        else:
            atom_positions = atom_positions_protein
        #print(atom_positions.shape)
            
        
        #print('This is channel ', atom_positions)

        atom_positions = torch.FloatTensor(atom_positions).to(device_type)
        

        # xgrid.view(-1, 1) is 125,000 long, because it's viewing a 50x50x50 cube in one column
        # then you repeat that column horizontally for each atom
        xx_xx = xgrid.view(-1, 1).repeat(1, len(atom_positions))
        yy_yy = ygrid.view(-1, 1).repeat(1, len(atom_positions))
        zz_zz = zgrid.view(-1, 1).repeat(1, len(atom_positions))
        
        # at this point we've created 3 arrays that are 125,000 long
        # and as wide as the number of atoms that are the current channel type
        # these 3 arrays just contain the flattened x,y,z positions of our 50x50x50 box


        # now do the same thing as above, just with the ACTUAL atomic position data
        posx_posx = atom_positions[:, 0].contiguous().view(1, -1).repeat(len(xgrid.view(-1)), 1)
        #print(xx_xx[0].shape)
        posy_posy = atom_positions[:, 1].contiguous().view(1, -1).repeat(len(ygrid.view(-1)), 1)
        posz_posz = atom_positions[:, 2].contiguous().view(1, -1).repeat(len(zgrid.view(-1)), 1)
        # three tensors of the same size, with actual atomic coordinates

        # normalizes the atomic positions with respect to the center of the box
        # and calculates density of atoms in each voxel
        
        bin_size = torch.tensor(float(bin_size)).to(device_type)
        sigma = 0.5*bin_size

        
        if channel in named_prop:
            prop = []
            for atom in protein_dict['smarts']:
    #atom.__getattribute__(prop)
                prop.append(atom.__getattribute__(named_prop[np.int(np.where(named_prop == channel)[0])]))
            
            if ligand == True:
                for atom in ligand_dict['smarts']:
    #atom.__getattribute__(prop)
                    prop.append(atom.__getattribute__(named_prop[np.int(np.where(named_prop == channel)[0])]))
                
            prop = np.array(prop)

            normalized_prop =  norm_properties(prop, channel)
            normalized_prop =  torch.FloatTensor(normalized_prop).to(device_type)
            #print(normalized_prop)
            
            density = torch.exp(-(((xx_xx - posx_posx)**2) * normalized_prop
                              + ((yy_yy - posy_posy)**2) * normalized_prop
                              + ((zz_zz - posz_posz)**2) * normalized_prop) / (2 * (sigma)**2)
                              )
        
        else:
            density = torch.exp(-((xx_xx - posx_posx)**2
                            + (yy_yy - posy_posy)**2
                            + (zz_zz - posz_posz)**2) / (2 * (sigma)**2))
        #print(density.shape)

        # Normalize so each atom density sums to one
        density /= torch.sum(density, dim=0)

        # Sum densities and reshape to original shape
        sum_densities = torch.sum(density, dim=1).view(xgrid.shape)
        #print(sum_densities.shape)

        # set all nans to 0
        sum_densities[sum_densities != sum_densities] = 0

        # add two empty dimmensions to make it 1x1x50x50x50, needed for CNN
        # sum_densities = sum_densities.unsqueeze(0)
        # sum_densities = sum_densities.unsqueeze(0)

        #fields[atom_type_index] = sum_densities
        fields[channel] = sum_densities.numpy()

#     if return_bins:
#         return fields, num_bins
#     else:
    return fields

def check_channel(channel, filter_set):
    """
    This function checks to see if a channel the user is asking to make a field
        for is an allowed channel to ask for.
    Parameters:
        channel (str, required): The atomic channel being requested
        filter_set (dict, required): The set of defined atomic filters
    Returns:
        boolean: indicator for if the channel is allowed
    """
    channel_allowed = False
    for key in filter_set:
        if channel in filter_set[key]:
            channel_allowed = True

    return channel_allowed

def find_channel_atoms(channel, protein_dict, filter_set):
    """
    This function finds the coordinates of all relevant atoms in a channel.
    It uses the filter set to constrcut the atomic channel (i.e., a channel can
        be composed of multiple filters).
    Parameters:
        channel (str, required): The atomic channel being constructed
        protein_dict (dict, required): The dictionary of the protein, returned from
            load_input()
        filter_set (dict, required): The set of available filters to construct channels with
    Returns:
        numpy array:  array containing the coordinates of each atom that is relevant
            to the channel
    """
    if channel in filter_set['atom']:
        atom_positions = protein_dict['shifted_positions'][protein_dict['atom_types'] == channel]
    
    
    elif channel in filter_set['general']:
        atom_dict = {'all_C' :'C', 'all_O' : 'O', 'all_N' : 'N'}
        atom_positions = protein_dict['shifted_positions']                        [[a.startswith(atom_dict[channel], 0) for a in protein_dict['atom_types']]]

    elif channel in filter_set['residue']:
        atom_positions = protein_dict['shifted_positions'][protein_dict['residues'] == channel]
        
    elif channel in filter_set['smarts_property']:
        smarts_list = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
        smarts_list_entry = smarts_list[np.int(np.where(filter_set['smarts_property'] == channel)[0])]
        pattern = pybel.Smarts(smarts_list_entry)
        atoms_smart = np.array(list(*zip(*pattern.findall(protein_dict['smarts']))),
                                    dtype=int) - 1
        #print(atoms_smart)
        atom_positions = protein_dict['shifted_positions'][atoms_smart]
    
    elif channel in filter_set['atom_property']:
        atom_positions = protein_dict['shifted_positions']
        
    elif channel in filter_set['protein_ligand']:
        atom_positions = protein_dict['shifted_positions']

    elif channel in filter_set['other']: # backbone or sidechain
        if channel == 'backbone':
            # create boolean arrays for backbone atoms
            bool_oxygen = protein_dict['atom_types'] == 'O'
            bool_carbon = protein_dict['atom_types'] == 'C'
            bool_alphacarbon = protein_dict['atom_types'] == 'CA'
            bool_nitrogen = protein_dict['atom_types'] == 'N'

            # sum of all the backbone channels into one boolean array
            bool_backbone = bool_oxygen + bool_carbon + bool_alphacarbon + bool_nitrogen

            # select the backbone atoms
            atom_positions = protein_dict['shifted_positions'][bool_backbone]

        else: # it was 'sidechain' filter, so grab sidechain atoms
            backbone_atom_set = np.array(['O', 'C', 'CA', 'N'])
            sidechain_atom_set = np.array([atom for atom in protein_dict['atom_type_set']                                            if atom not in backbone_atom_set])

            for index, sidechain_atom in enumerate(sidechain_atom_set):
                if index == 0:
                    # create the first sidechains boolean array, will be edited
                    bool_sidechains = protein_dict['atom_types'] == sidechain_atom
                else:
                    # single boolean array for the current sidechain atom
                    bool_atom = protein_dict['atom_types'] == sidechain_atom

                    # sum this boolean array with the master boolean array
                    bool_sidechains += bool_atom

            # grab all sidechain atom positions
            atom_positions = protein_dict['shifted_positions'][bool_sidechains]

    else: # it was a residue property channel
        acidic_residues = np.array(['ASP', 'GLU'])
        basic_residues = np.array(['LYS', 'ARG', 'HIS'])
        polar_residues = np.array(['GLN', 'ASN', 'HIS', 'SER', 'THR', 'TYR', 'CYS'])
        nonpolar_residues = np.array(['GLY', 'ALA', 'VAL', 'LEU',                                         'ILE', 'MET', 'PRO', 'PHE', 'TRP'])
        amphipathic_residues = np.array(['TRP', 'TYR', 'MET'])
        charged_residues = np.array(['ARG', 'LYS', 'ASP', 'GLU'])
        # custom_residues = something
        property_dict = {'acidic':acidic_residues, 'basic':basic_residues,                         'polar':polar_residues, 'nonpolar':nonpolar_residues,                         'amphipathic':amphipathic_residues, 'charged':charged_residues}

        atom_positions = atoms_from_residues(protein_dict, property_dict[channel])

    return atom_positions

def atoms_from_residues(protein_dict, residue_list):
    """
    This function finds all the atoms in a protein that are members of any residues
    in a list of residues.
    Parameters:
        protein_dict (dict, required): The dictionary of the protein, returned from
            load_input()
        residue_list (list-like, required): The list of residues whose atoms we are
            finding coordinates for
    """
    # construct the appropriate boolean array to index the atoms in the protein_dict
    for index, residue in enumerate(residue_list):
        if index == 0:
            bool_residue = protein_dict['residues'] == residue
        else:
            bool_residue += protein_dict['residues'] == residue

    atom_positions = protein_dict['shifted_positions'][bool_residue]

    return atom_positions

def voxelize(path, channels=['CA'], path_type='file', ligand=False, bin_size=2.0, num_bins=50, save=False, save_fn='voxels.npy', save_path='./'):
    """
    This function creates a dictionary of tensor fields directly from a pdb file.
    These tensor fields can be plotted, or sent directly into the cnn for
        plotting internals, or sent all the way through a cnn/vae to be used for
        training.
    Parameters:
        path (str, required): path to a .pdb file
        channels (list of strings, optional): The list of atomic channels to be included in
            the output dictionary, one field for every channel.
            Any channels from points 1-4 below may be combined in any order.
            i.e., one could call voxelize with the channels parameter as
            channels=['charged', 'CB', 'GLY', 'polar', ...etc]. Note that voxelization
            for channels containing more atoms will take longer.
            1. any of the following atom types
                ['C' 'CA' 'CB' 'CD' 'CD1' 'CD2' 'CE' 'CE1' 'CE2' 'CE3' 'CG' 'CG1' 'CG2'
                'CH2' 'CZ' 'CZ2' 'CZ3' 'N' 'ND1' 'ND2' 'NE' 'NE1' 'NE2' 'NH1' 'NH2' 'NZ'
                'O' 'OD1' 'OD2' 'OE1' 'OE2' 'OG' 'OG1' 'OH' 'OXT' 'SD' 'SG']
            2. Any canonical residue in the protein, using the three letter residue code, all caps
               (NOTE: the residue must actually exist in the protein)
                e.g., ['LYS', 'LEU', 'ALA']
            3. The 'other' channel options: 'backbone', 'sidechains'
            4. There are 6 channels corresponding to specific types of residues:
                'charged', 'polar', 'nonpolar', 'amphipathic', 'acidic', 'basic'
    Returns:
        dictionary: a dictionary containing a voxelized atomic fields, one for each
        channel requested. Each field has shape = ([1, 1, 50, 50, 50])
    """
    if path_type == 'file':
        pro_dict = load_input(path, ligand=ligand)
        if ligand == True:
            protein_dict = pro_dict[0]
            ligand_dict = pro_dict[1]
            return make_fields(protein_dict, channels=channels, bin_size=bin_size, num_bins=num_bins, ligand=True, ligand_dict=ligand_dict)
        else:
            protein_dict = pro_dict
            sys.stdout.write('done')
            return make_fields(protein_dict, channels=channels, bin_size=bin_size, num_bins=num_bins)

    # ------------FOLDER FUNCTIONALITY INCOMPLETE---------------
    elif path_type == 'folder':
        fields = []
        pdb_fns = os.listdir(path)
        for j, fn in enumerate(pdb_fns):
            progress = '{}/{} pdbs voxelized ({}%)'.format(j, len(pdb_fns),                         round(j / len(pdb_fns) * 100, 2))
            sys.stdout.write('\r'+progress)
            protein_dict = load_input(os.path.join(path, fn))
            field, bins = make_fields(protein_dict, channels=channels, bin_size=bin_size, num_bins=num_bins)
            channel_list = []
            for channel in channels:
                channel_list.append(field[channel].reshape(1,
                                                           bins,
                                                           bins,
                                                           bins))
            field = np.concatenate(channel_list).reshape(1,len(channels),
                                                         bins, bins, bins)
            fields.append(field)
        sys.stdout.write("\r\033[K")
        out_statement = 'voxelization complete!\n'
        sys.stdout.write('\r'+out_statement)
        fields = np.concatenate(fields)

        if save:
            np.save(os.path.join(save_path, save_fn), fields)
        else:
            return fields


# In[9]:


file_name = '1a1e_pocket.mol2'
myprotein_dict = load_input(file_name, ligand=True)
myprotein_dict[0]


# In[10]:


# channel_list = ['all_C', 'all_O', 'all_N', 'acidic', 'basic', 'polar', 'nonpolar',\
#                 'charged', 'amphipathic','hydrophobic', 'aromatic', 'acceptor', 'donor',\
#                 'ring', 'hyb', 'heavyvalence', 'heterovalence', 'partialcharge','protein', 'ligand']
channel_list = ['all_O', 'all_C']
#other_filters = np.array(['backbone', 'sidechains'])
voxel = voxelize(file_name, channels=channel_list, bin_size=2.0,num_bins=50, ligand=True)


# In[ ]:


@click.command()
@click.option('--input_dir', help='input directory to search for pocket.mol2 files')
@click.option('--output_dir', help='output directory to save the computed data')
def run(input_dir, output_dir):
    """runs the tool over all pdbs in a directory"""
    for input_filename in os.listdir(input_dir):
        output_filename = os.path.join(output_dir, input_filename + '.csv')
        if input_filename.endswith(".pdb") and not os.file.exists(output_filename):
            do_all_the_work(input_filename, output_filename)
        else:
            continue
    return


# In[52]:

