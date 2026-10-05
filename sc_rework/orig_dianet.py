import torch
import torch.nn as nn

class Script_DiaNet(nn.Module):

    def __init__(self, inp_num, out_num, device='cpu'):
        super(Script_DiaNet, self).__init__()
        self.log = False
        
        self.red_depths = list(range(2, 1000))
        self.red_fulls = [sum(range(1, n+2)) for n in self.red_depths]
        self.red_heads = [(x * 2) + 1 for x in self.red_depths]

        self.red_full = -1
        self.red_dep = -1
        self.red_triangle = []
        self.jump_weights = nn.ParameterList()
        self.layers, self.masks = self.gen_layers(inp_num, out_num)
        # each mask to device
        for i in range(len(self.masks)):
            self.masks[i] = self.masks[i].to(device)
        # each layers multiply mask
        self.x_hidden = []
        for i in range(len(self.layers)):
            self.layers[i].weight.data = self.layers[i].weight.data * self.masks[i]
            self.x_hidden.append(-1) # init x_hidden
        
    
    def select_insert(self, n, sel_len) -> list:
        assert (((n - 1) // 2) % 2) == (sel_len % 2), "Invalid selection length"
        numbers = range(n)
        odd_numbers = [num for num in numbers if num % 2 == 1]
        center_index = len(odd_numbers) // 2
        selected_numbers = odd_numbers[center_index - sel_len // 2: center_index + sel_len // 2 + (sel_len % 2)]
        return selected_numbers if sel_len != 1 else selected_numbers
        
    def gen_layers(self, inp_num, out_num):
        # create Red triangle(input triangle)
        masks_list = []
        self.bn_list = [] # batch norm list

        # select input triangle by input num
        red_dep, red_head = -1, -1
        for i in range(len(self.red_fulls)-1):
            if inp_num > self.red_fulls[i] and inp_num <= self.red_fulls[i+1]:
                red_full = self.red_fulls[i+1]
                red_dep = self.red_depths[i+1]
                red_head = self.red_heads[i+1]
                if self.log: print("red_dep:", red_dep, "red_head:", red_head)
                break
        assert (red_dep>0)and(red_head>0), "red_dep {} or head {} is not valid".format(red_dep, red_head)
        self.red_dep = red_dep
        self.red_full = red_full
        for i in range(red_dep, 0, -1):
            if i == red_dep:
                self.red_triangle.append(i*2+1)
            else:
                self.red_triangle.append(i)

        
        # create blue triangle(output triangle)
        nn_output = out_num + (out_num-1)
        blue_dep = int(nn_output/2-0.5)
        if self.log:print("blue_dep:", blue_dep)

        total_dep = red_dep + blue_dep
        if self.log:print("total_dep:", total_dep)

        # expand times
        expand_times = total_dep - red_dep
        if self.log:print("expand_times:", expand_times)

        # create expand time nn layer
        expand_list = []
        expand_list.append(nn.Linear(red_head, red_head-2, bias=False)) # first layer
        self.bn_list.append(nn.BatchNorm1d(red_head-2))
        self.jump_weights.append(nn.Parameter(torch.randn(1, red_head-2))) # Not in use

        masks_list.append(self.fst_mask(red_head, red_head-2))
        nn_in = red_head-2
        for i in range(expand_times):
            expand_list.append(nn.Linear(nn_in, nn_in+2, bias=False))
            self.bn_list.append(nn.BatchNorm1d(nn_in+2))
            masks_list.append(self.exp_mask_v2(nn_in, nn_in+2))
            self.jump_weights.append(nn.Parameter(torch.randn(1, nn_in-2)))
            nn_in += 2
        if self.log:print("expand_list:", expand_list)

        # shrink times
        shrink_times = total_dep - blue_dep - 1
        # if self.log:print("shrink_times:", shrink_times)
        shrink_list = []
        for i in range(shrink_times):
            shrink_list.append(nn.Linear(nn_in, nn_in-2, bias=False))
            self.bn_list.append(nn.BatchNorm1d(nn_in-2))
            masks_list.append(self.shr_mask_v2(nn_in, nn_in-2))
            self.jump_weights.append(nn.Parameter(torch.randn(1, nn_in-2)))
            nn_in -= 2
        if self.log:print("shrink_list:", shrink_list)

        # combine expand and shrink list
        layer_list = expand_list + shrink_list
        layers = nn.Sequential(*layer_list)
        assert len(masks_list) == len(layer_list), "masks_list {} and layer_list {} is not same".format(len(masks_list), len(layer_list))

        return layers, masks_list
    
    def fst_mask(self, in_dim, out_dim):
        assert in_dim-out_dim == 2, "fst_mask: in_dim {} and out_dim {} is not valid".format(in_dim, out_dim)
        mask = torch.zeros((out_dim, in_dim))
        for i in range(out_dim):
            if i%2 == 0:
                start_idx = i
                end_idx = i+2 + 1
                mask[i, start_idx:end_idx] = 1
        return mask
    
    def exp_mask(self, in_dim, out_dim):
        assert out_dim-in_dim == 2, "exp_mask: in_dim {} and out_dim {} is not valid".format(in_dim, out_dim)
        mask = torch.zeros((out_dim, in_dim))
        for i in range(out_dim):
            if i%2 == 0:
                a = max(0, i-4)
                b = min(max(0, i-2), in_dim-1)
                c = min(max(0, i-1), in_dim-1)
                d = min(max(0, i), in_dim-1)
                e = min(max(0, i+2), in_dim-1)
                idx_lis = [a,b,c,d,e]
                mask[i, idx_lis] = 1
        return mask
    
    def shr_mask(self, in_dim, out_dim):
        assert in_dim-out_dim == 2, "shr_mask: in_dim {} and out_dim {} is not valid".format(in_dim, out_dim)
        mask = torch.zeros((out_dim, in_dim))
        for i in range(out_dim):
            if i%2 == 0:
                a = max(0, i-2)
                b = min(max(0, i), in_dim-1)
                c = min(max(0, i+1), in_dim-1)
                d = min(max(0, i+2), in_dim-1)
                e = min(max(0, i+4), in_dim-1)
                idx_lis = [a,b,c,d,e]
                mask[i, idx_lis] = 1
        return mask

    def exp_mask_v2(self, in_dim, out_dim):
        assert out_dim-in_dim == 2, "exp_mask: in_dim {} and out_dim {} is not valid".format(in_dim, out_dim)
        mask = torch.zeros((out_dim, in_dim))
        for i in range(out_dim):
            if i%2 == 0:
                a = max(0, i-6)
                b = min(max(0, i-4), in_dim-1)
                c = min(max(0, i-2), in_dim-1)
                d = min(max(0, i-1), in_dim-1)
                e = min(max(0, i), in_dim-1)
                f = min(max(0, i+2), in_dim-1)
                g = min(max(0, i+4), in_dim-1)
                idx_lis = [a,b,c,d,e,f,g]
                mask[i, idx_lis] = 1
        return mask

    def shr_mask_v2(self, in_dim, out_dim):
        assert in_dim-out_dim == 2, "shr_mask: in_dim {} and out_dim {} is not valid".format(in_dim, out_dim)
        mask = torch.zeros((out_dim, in_dim))
        for i in range(out_dim):
            if i%2 == 0:
                a_ = max(0, i-4)
                a = min(max(0, i-2), in_dim-1)
                b = min(max(0, i), in_dim-1)
                c = min(max(0, i+1), in_dim-1)
                d = min(max(0, i+2), in_dim-1)
                e = min(max(0, i+4), in_dim-1)
                e_ = min(max(0, i+6), in_dim-1)
                idx_lis = [a_,a,b,c,d,e,e_]
                mask[i, idx_lis] = 1
        return mask
                

    def forward(self, x):
        container = torch.zeros((x.shape[0], self.red_full)) # (b, 28)
        container[:, 0:x.shape[1]] = x
        # split x for red triangle, froexample, 25-> 13,5,4,3,2,1
        x_seg = []
        slice_idx = 0
        for i in range(self.red_dep, 0, -1):
            if i == self.red_dep:
                x_seg.append(container[:, 0:i*2+1]) # 13
                slice_idx += i*2+1
            else:
                x_seg.append(container[:, slice_idx:slice_idx+i]) # 13:18, 18:22, 22:25, 25:27, 27:28
                slice_idx += i
        if self.log: 
            for seg in x_seg: print(seg)
        if self.log: print('*'*30)

        for i, layer in enumerate(self.layers):
            # red triangle stage
            if i == 0:
                layer.weight.data = torch.clamp(layer.weight.data, min=-1, max=1)
                layer.weight.data *= self.masks[i]
                x = layer(x_seg[i])
                x = torch.tanh(x)
                # x = self.bn_list[i](x)
                if False: write_debug('layer{}, x'.format(i), x.detach().numpy())
                self.x_hidden[i] = x
            
            elif i == 1:
                if False: write_debug('layer{}, x'.format(i), x.detach().numpy())
                assert torch.sum(x[:, 1::2]) == 0, "x[:, 1::2] is not 0"
                layer.weight.data = torch.clamp(layer.weight.data, min=-1, max=1)
                layer.weight.data *= self.masks[i]
                x_cpy = x.clone()
                this_len, red_tri_layer_num = x_cpy.shape[1], self.red_triangle[i]
                sel_insert = self.select_insert(this_len, red_tri_layer_num)
                # if self.log: print("sel_insert:", sel_insert)
                assert x_cpy[:, sel_insert].shape == x_seg[i].shape, "x_cpy[:, sel_insert] shape {} and x_seg[i] shape {} is not same".format(x_cpy[:, sel_insert].shape, x_seg[i].shape)
                if False: write_debug('layer{}, seg'.format(i), x_seg[i].detach().numpy())
                x_cpy[:, sel_insert] += x_seg[i]
                x = x_cpy
                if False: write_debug('layer{}, x insrt'.format(i), x.detach().numpy())
                x = layer(x)
                x = torch.tanh(x)
                # x = self.bn_list[i](x)
                self.x_hidden[i] = x

            elif i > 1 and i < self.red_dep:
                if False: write_debug('layer{}, x'.format(i), x.detach().numpy())
                assert torch.sum(x[:, 1::2]) == 0, "x[:, 1::2] is not 0"
                layer.weight.data = torch.clamp(layer.weight.data, min=-1, max=1)
                layer.weight.data *= self.masks[i]
                x_cpy = x.clone()
                this_len = x_cpy.shape[1]
                red_tri_layer_num = self.red_triangle[i]
                sel_insert = self.select_insert(this_len, red_tri_layer_num)
                # if self.log: print("sel_insert:", sel_insert)
                assert x_cpy[:, sel_insert].shape == x_seg[i].shape, "x_cpy[:, sel_insert] shape {} and x_seg[i] shape {} is not same".format(x_cpy[:, sel_insert].shape, x_seg[i].shape)
                if False: write_debug('layer{}, seg'.format(i), x_seg[i].detach().numpy())
                # insert input segment to x_cpy
                x_cpy[:, sel_insert] += x_seg[i]
                x = x_cpy
                if False: write_debug('layer{}, x insrt'.format(i), x.detach().numpy())
                x = layer(x)
                if False: write_debug('layer{}, x fc'.format(i), x.detach().numpy())
                # jump connection, add prevprev layer output to this layer output
                jump = self.x_hidden[i-2]
                if False: write_debug('layer{}, jump'.format(i), jump.detach().numpy())
                assert torch.sum(jump[:, 1::2]) == 0, "jump[:, 1::2] is not 0"
                assert (jump.shape[1] == x.shape[1]+4) or (jump.shape[1] == x.shape[1]-4) or (jump.shape[1] == x.shape[1]), "jump.shape[1] {} and x.shape[1] {} is not valid".format(jump.shape[1], x.shape[1])
                x_cpy = x.clone()
                # compare x_cpy and jump
                # jump_w = self.jump_weights[i]
                # jump_w = torch.clamp(jump_w, min=-1, max=1)
                if x_cpy.shape[1] > jump.shape[1]:
                    x_cpy[:, 2:-2] += jump #* jump_w
                elif x_cpy.shape[1] == jump.shape[1]:
                    x_cpy += jump #* jump_w
                else:
                    x_cpy += jump[:, 2:-2] #* jump_w
                x = x_cpy
                if False: write_debug('layer{}, jump_w'.format(i), self.jump_weights[i].detach().numpy())
                if False: write_debug('layer{}, x jump'.format(i), x.detach().numpy())
                x = torch.tanh(x)
                # x = self.bn_list[i](x)
                self.x_hidden[i] = x

            elif i >= self.red_dep:
                if False: write_debug('layer{}, x'.format(i), x.detach().numpy())
                assert torch.sum(x[:, 1::2]) == 0, "x[:, 0::2] is not 0"
                layer.weight.data = torch.clamp(layer.weight.data, min=-1, max=1)
                layer.weight.data *= self.masks[i]
                x = layer(x)
                if False: write_debug('layer{}, x fc'.format(i), x.detach().numpy())
                jump = self.x_hidden[i-2]
                if False: write_debug('layer{}, jump'.format(i), jump.detach().numpy())
                assert torch.sum(jump[:, 1::2]) == 0, "jump[:, 1::2] is not 0"
                assert (jump.shape[1] == x.shape[1]+4) or (jump.shape[1] == x.shape[1]-4) or (jump.shape[1] == x.shape[1]), "jump.shape[1] {} and x.shape[1] {} is not valid".format(jump.shape[1], x.shape[1])
                x_cpy = x.clone()
                # compare x_cpy and jump
                # jump_w = self.jump_weights[i]
                # jump_w = torch.clamp(jump_w, min=-1, max=1)
                if x_cpy.shape[1] > jump.shape[1]:
                    x_cpy[:, 2:-2] += jump #* jump_w
                elif x_cpy.shape[1] == jump.shape[1]:
                    x_cpy += jump #* jump_w
                else:
                    x_cpy += jump[:, 2:-2] #* jump_w
                x = x_cpy
                if False: write_debug('layer{}, jump_w'.format(i), self.jump_weights[i].detach().numpy())
                if False: write_debug('layer{}, x jump'.format(i), x.detach().numpy())
                x = torch.tanh(x)
                # x = self.bn_list[i](x)
                self.x_hidden[i] = x
            
        # select all the even index of x
        x = x[:, 0::2]

        return x
            
            

