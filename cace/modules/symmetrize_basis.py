import torch

__all__ = ['symmetrize_A_basis']

def symmetrize_A_basis(nu_max: int, 
                       vec_dict_allnu: dict,
                       node_attr: torch.Tensor, 
                       l_list: list):
    """
    Symmetrize the node attributes for the A basis,
    to make the B basis.
    """
    num_nodes, n_radial, n_angular, n_chanel = node_attr.size()
    sym_node_attr = {}

    for nu in range(1, nu_max+1):
        if nu == 1:
            sym_node_attr[nu] = torch.zeros((num_nodes, n_radial, 1, n_chanel))
            sym_node_attr[nu][:,:,0,:] = node_attr[:,:,0,:]
            #print(sym_node_attr[nu].shape)

        if nu == 2:
            vec_dict = vec_dict_allnu[2]
            sym_node_attr[nu] = torch.zeros((num_nodes, n_radial, len(vec_dict), n_chanel))

            for i, (l_now, lxlylz_list) in enumerate(vec_dict.items()):
                #print(l_now)
                for (lxlylz1, prefactor) in lxlylz_list:
                    index_1 = l_list.index(lxlylz1)
                    sym_node_attr[nu][:,:,i,:] += prefactor * \
                        node_attr[:,:,index_1,:] * node_attr[:,:,index_1,:]
            #print(sym_node_attr[nu].shape)

        if nu == 3:
            vec_dict = vec_dict_allnu[3]
            sym_node_attr[nu] = torch.zeros((num_nodes, n_radial, len(vec_dict), n_chanel))

            for i, (l_now, lxlylz_list) in enumerate(vec_dict.items()):
                #print(l_now)
                for (lxlylz1, lxlylz2, lxlylz3, prefactor) in lxlylz_list:
                    index_1 = l_list.index(lxlylz1)
                    index_2 = l_list.index(lxlylz2)
                    index_3 = l_list.index(lxlylz3)
                    sym_node_attr[nu][:,:,i,:] += prefactor * \
                        node_attr[:,:,index_1,:] * node_attr[:,:,index_2,:] * node_attr[:,:,index_3,:]
            #print(sym_node_attr[nu].shape)

        if nu == 4:
            vec_dict = vec_dict_allnu[4]
            sym_node_attr[nu] = torch.zeros((num_nodes, n_radial, len(vec_dict), n_chanel))

            for i, (l_now, lxlylz_list) in enumerate(vec_dict.items()):
                for (lxlylz1, lxlylz2, lxlylz3, lxlylz4, prefactor) in lxlylz_list:
                    index_1 = l_list.index(lxlylz1)
                    index_2 = l_list.index(lxlylz2)
                    index_3 = l_list.index(lxlylz3)
                    index_4 = l_list.index(lxlylz4)
                    sym_node_attr[nu][:,:,i,:] += prefactor * \
                        node_attr[:,:,index_1,:] * node_attr[:,:,index_2,:] * \
                        node_attr[:,:,index_3,:] * node_attr[:,:,index_4,:]
            #print(sym_node_attr[nu].shape)

        if nu >= 5:
            raise NotImplementedError

    return torch.cat(list(sym_node_attr.values()), dim=2) 
