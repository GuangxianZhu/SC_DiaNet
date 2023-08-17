from tabulate import tabulate



#from tabulate import tabulate

#基本完工的函数

# 生成输入的拓扑
def create_input_site(attrb_num):
    #探测所需深度
    layer_num = 0
    input_site_num = 0
    while (input_site_num < attrb_num):
        layer_num += 1
        input_site_num += layer_num

    #根据深度生成倒置三角形
    input_triangle_width = [i for i in range(layer_num, 0, -1)]

    input_site_array = []
    input_site_width = []
    base_site = 0

    for num in input_triangle_width:
        row = list(range(base_site, base_site + num))
        
        if base_site + num >= attrb_num:
           row = row[:attrb_num - base_site]

        input_site_array.append(row)
        input_site_width.append(len(row))
        base_site += num
    
        if base_site >= attrb_num:
            break

    return input_site_array, input_site_width

#创建神经元拓扑
def create_dianet_neuron_array(input_site_array, label_num):

    #各层的神经元数量
    dianet_layer_width = []
    
    #extend_layer 生成
    dianet_layer_width.append(len(input_site_array[0]))
    for i in range(label_num - 1):
        dianet_layer_width.append(dianet_layer_width[i] + 1)

    #compress_layer 生成
    layer_cnt = dianet_layer_width[-1]
    while(layer_cnt > label_num):
        layer_cnt -= 1
        dianet_layer_width.append(layer_cnt)

    #OK, 我们得到了dianet的各层宽度 dianet_layer_width
    #生成实际的拓扑
    dianet_neuron_array = []

    for i in range(len(dianet_layer_width)):
        dianet_temp_layer = []
        for j in range(dianet_layer_width[i]):
            dianet_temp_layer.append(j)
        dianet_neuron_array.append(dianet_temp_layer)

    return dianet_neuron_array, dianet_layer_width


# 神经元属性的字典
neuron_unit = {
    'layer': None, 'index': None, 'type': None, 'area': None, 'locate': None, 
    'left_i': None, 'right_i': None, 'left_o': None, 'right_o': None, 
    'tap_i': None, 

    'ly-3_l2': None,
    'ly-3_l1': None,
    'ly-3_r1': None,
    'ly-3_r2': None,
    'ly-2_l1': None,
    'ly-2_m': None,
    'ly-2_r1': None,
    'ly-1_l1': None,
    'ly-1_r1': None,
    'skip_i': None, 'skip_o': None
}

#创建dianet的属性列表
def create_dianet_neuron_feature(input_site_array, dianet_neuron_array):
    dianet_neuron_feature = []
    label_num = len(dianet_neuron_array[-1])


    for i, layer in enumerate(dianet_neuron_array):
        layer_neuron_feature = []

        for j in layer:
            #开始合成神经元的条件
            n_type = None
            n_locate = None
            n_area = None
            n_left_i = None
            n_right_i = None
            n_tap_i = None
            n_skip_i = None
            n_left_o = None
            n_right_o = None
            n_skip_o = None

            #确定类型
            if i == 0: 
                n_type = 'in'
            elif i == len(dianet_neuron_array) - 1:
                n_type = 'out'
            else:
                n_type = 'hid'

            #确定位置
            if j == 0:
                n_locate = 'leftend'
            elif j == len(layer) - 1:
                n_locate = 'rightend'
            else:
                n_locate = 'middle'

            #确定区域形状
            if i == 0: #排除边界错误
                if len(layer) - len(dianet_neuron_array[i+1]) == 1:
                    n_area = 'compress_area'
                else:
                    n_area = 'extend_area'

            elif (0 < i < len(dianet_neuron_array) - 1) and \
                (len(dianet_neuron_array[i-1]) == len(dianet_neuron_array[i+1])):
                n_area = 'max_layer'


            else:
                if len(layer) - len(dianet_neuron_array[i-1]) == 1:
                    n_area = 'extend_area'
                else:
                    n_area = 'compress_area'


            #标准输入输出网表
            #n_left_i
            if n_type == 'in':
                n_left_i = None
            elif (n_area == 'extend_area') and (n_locate == 'leftend'):
                n_left_i = None
            elif (n_area == 'max_layer') and (n_locate == 'leftend'):
                n_left_i = None
            else:
                if n_area == 'extend_area':
                    n_left_i = [[i-1, j-1], [i, j]]
                elif n_area == 'max_layer':
                    n_left_i = [[i-1, j-1], [i, j]]
                elif n_area == 'compress_area':
                    n_left_i = [[i-1, j], [i, j]]
                else:
                    print("Dianet's topolog error!n_left_i")

            #n_right_i
            if n_type == 'in':
                n_right_i = None
            elif (n_area == 'extend_area') and (n_locate == 'rightend'):
                n_right_i = None
            elif (n_area == 'max_layer') and (n_locate == 'rightend'):
                n_right_i = None
            else:
                if n_area == 'extend_area':
                    n_right_i = [[i-1, j], [i, j]]
                elif n_area == 'max_layer':
                    n_right_i = [[i-1, j], [i, j]]
                elif n_area == 'compress_area':
                    n_right_i = [[i-1, j+1], [i, j]]
                else:
                    print("Dianet's topolog error!n_right_i")

            #n_left_o
            if n_type == 'out':
                n_left_o = None
            elif (n_area == 'compress_area') and (n_locate == 'leftend'):
                n_left_o = None
            elif (n_area == 'max_layer') and (n_locate == 'leftend'):
                n_left_o = None
            else:
                if n_area == 'extend_area':
                    n_left_o = [[i, j], [i+1, j]]
                elif n_area == 'max_layer':
                    n_left_o = [[i, j], [i+1, j-1]]
                elif n_area == 'compress_area':
                    n_left_o = [[i, j], [i+1, j-1]]
                else:
                    print("Dianet's topolog error!n_left_o")

            #n_right_o
            if n_type == 'out':
                n_right_o = None
            elif (n_area == 'compress_area') and (n_locate == 'rightend'):
                n_right_o = None
            elif (n_area == 'max_layer') and (n_locate == 'rightend'):
                n_right_o = None
            else:
                if n_area == 'extend_area':
                    n_right_o = [[i, j], [i+1, j+1]]
                elif n_area == 'max_layer':
                    n_right_o = [[i, j], [i+1, j]]
                elif n_area == 'compress_area':
                    n_right_o = [[i, j], [i+1, j]]
                else:
                    print("Dianet's topolog error!n_right_o")


            #n_tap_i
            #算法是, 当层索引i < label_num时, input_site_offset和当前i值相等, i >= label_num时, offset固定为label_num - 1
            if i < label_num:
                input_site_offset = i
            else:
                input_site_offset = label_num - 1


            if (i < len(input_site_array)) and (j >= input_site_offset) and ((j - input_site_offset) < len(input_site_array[i])):
                n_tap_i = input_site_array[i][j - input_site_offset]

            #n_skip_i
            if len(dianet_neuron_array) >= 4: #此时才存在跳层
                if (i >= 3) and (i <= (len(dianet_neuron_array)-1)):
                    if (len(layer) - len(dianet_neuron_array[i-2]) == 0):
                        n_skip_i = [[i-2, j], [i, j]]

                    elif (len(layer) - len(dianet_neuron_array[i-2]) == 2):
                        if j == 0:
                            n_skip_i = None
                        elif j == len(layer) - 1:
                            n_skip_i = None
                        else:
                            n_skip_i = [[i-2, j-1], [i, j]]

                    elif (len(layer) - len(dianet_neuron_array[i-2]) == -2):
                        n_skip_i = [[i-2, j+1], [i, j]]

                    else:
                        print("Dianet's topolog error!")
            else:
                n_skip_i = None

            #n_skip_o
            if len(dianet_neuron_array) >= 4: #此时才存在跳层
                if (i >= 1) and (i <= (len(dianet_neuron_array)-3)):
                    if (len(layer) - len(dianet_neuron_array[i+2]) == 0):
                        n_skip_o = [[i, j], [i+2, j]]

                    elif (len(layer) - len(dianet_neuron_array[i+2]) == 2):
                        if j == 0:
                            n_skip_o = None
                        elif j == len(layer) - 1:
                            n_skip_o = None
                        else:
                            n_skip_o = [[i, j], [i+2, j-1]]

                    elif (len(layer) - len(dianet_neuron_array[i+2]) == -2):
                        n_skip_o = [[i, j], [i+2, j+1]]

                    else:
                        print("Dianet's topolog error!")
            else:
                n_skip_o = None


            neuron_unit = {
                'layer': i, 'index': j, 'type': n_type, 'locate': n_locate, 'area': n_area,
                'left_i': n_left_i, 'right_i': n_right_i, 'left_o': n_left_o, 'right_o': n_right_o,
                'tap_i': n_tap_i, 'skip_i': n_skip_i, 'skip_o': n_skip_o
            }

            layer_neuron_feature.append(neuron_unit)
            #print(neuron_unit)

        dianet_neuron_feature.append(layer_neuron_feature)
    
    return dianet_neuron_feature


#用于生成任意形状dianet, 支持超大型拓扑
def print_dianet_topolog(input_site_array, dianet_neuron_array):
    print('This is the topolog: ')

    dianet_longest_row = max(dianet_neuron_array, key=len)
    dianet_longest_element = dianet_longest_row[-1]

    input_shortest_row = min(input_site_array, key=len)
    input_longest_row = max(input_site_array, key=len)
    input_longest_element = input_shortest_row[-1]
    std_element_width = max(len(str(input_longest_element)), len(str(dianet_longest_element)))


    #max_element_width = max(len(str(num)) for row in data for num in row)

    for layer in dianet_neuron_array:
        spc_padding = (' ' * std_element_width) * (len(dianet_longest_row) - len(layer))
        elements = (' ' * std_element_width).join(f'{num:>{std_element_width}}' for num in layer)
        print(spc_padding + elements)



    print('\nThis is input site: ')

    input_spc_offset = len(dianet_longest_row) - len(input_longest_row)

    for layer in input_site_array:
        spc_padding = (' ' * std_element_width) * (input_spc_offset)
        elements = (' ' * std_element_width).join(f'{num:>{std_element_width}}' for num in layer)
        input_spc_offset += 1
        print(spc_padding + elements)


# 按表格打印神经元的属性
def print_dianet_neuron_feature(dianet_neuron_feature):

    table_headers = ["layer", "index", "type", "locate", "area", "left_i", "right_i", "left_o", "right_o", "tap_i", "skip_i", "skip_o"]
    
    table_rows = []
    for layer_data in dianet_neuron_feature:
        for item in layer_data:
            row = [item['layer'], item['index'], item['type'], item['locate'], item['area'],
                   item['left_i'], item['right_i'], item['left_o'], item['right_o'],
                   item['tap_i'], item['skip_i'], item['skip_o']]
            table_rows.append(row)
        table_rows.append('------------')
    table = tabulate(table_rows, headers=table_headers)
    print(table)