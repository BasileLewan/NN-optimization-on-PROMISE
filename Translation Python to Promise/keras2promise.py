

def parse_layer(nb, path, size):
    out = f"// Layer {nb} weights\n"
    if nb > 0:
        f = open(f"{path}dense_{nb}_kernel_array.csv", 'r')
    else:
        f = open(f"{path}dense_kernel_array.csv", 'r')

    w_list = f.read().splitlines()

    f.close()

    if size[0] == 1 or size[1] == 1:
        for i, w in enumerate(w_list):
            out += f"__PR_L{nb}N{i}__ l{nb}n{i}_w = {w};\n"
    elif size[0] * size[1] < 1e4:  # we can import weights in only one table if there are not many, else it will cause CADNA to crash
        out += f"double w_loader_{nb}[{size[0] * size[1]}] = {{"
        for w in w_list:
            out += w + ','
        out += '};\n'
        for i in range(size[1]):
            out += f"__PR_L{nb}N{i}__ l{nb}n{i}_w[{size[0]}];\n"
        out += f"\nfor (unsigned i = 0; i < {size[0] * size[1]}; i+={size[1]}) {{\n"
        for i in range(size[1]):
            out += f"l{nb}n{i}_w[i/{size[1]}] = w_loader_{nb}[i+{i}];\n"
        out += f"}}\n //free(w_loader_{nb});\n"
#     else:
#         w_load = [[] for _ in range(size[1])]
#         for i, w in enumerate(w_list):
#             w_load[i % size[1]].append(float(w[:11]))
#         for i in range(size[1]):
#             out += f"double w_loader_l{nb}n{i}[{size[0]}] = {{ {str(w_load[i])[1:-1] } }};\n "
#         for i in range(size[1]):
#             out += f"__PR_L{nb}N{i}__ l{nb}n{i}_w[{size[0]}];\n"
#         out += f"\nfor (unsigned i = 0; i < {size[0]}; i++) {{\n"
#         for i in range(size[1]):
#             out += f"l{nb}n{i}_w[i] = w_loader_l{nb}n{i}[i];\n"
#         out += f"}}\n"
#    else:
#        if nb > 0:
#            out += f"FILE* fd = fopen(\"{path}dense_{nb}_kernel_array.csv\", \"r\");\n"
#        else:
#            out += f"FILE* fd = fopen(\"{path}dense_kernel_array.csv\", \"r\");\n"
#        for i in range(size[1]):
#            out += f"__PR_L{nb}N{i}__* l{nb}n{i}_w = (__PR_L{nb}N{i}__*)malloc({size[0]} * sizeof(__PR_L{nb}N{i}__));\n"
#        out += f"double tmp;\nfor(unsigned k = 0; k < {size[0]}; k++){{\n"
#        for i in range(size[1]):
#            out += f"fscanf(fd, \"%lf\", &tmp); l{nb}n{i}_w[k] = tmp;\n"
#        out += "}\nfclose(fd);\n"
    else:
        for i in range(size[1]):
            out += f"__PR_L{nb}N{i}__* l{nb}n{i}_w = (__PR_L{nb}N{i}__*)malloc({size[0]} * sizeof(__PR_L{nb}N{i}__));\n"
        w_load = [[] for _ in range(size[1])]
        for i, w in enumerate(w_list):
            w_load[i % size[1]].append(float(w))
        for i in range(size[1]):
            out += f"{{\ndouble w_loader_l{nb}n{i}[{size[0]}] = {{ {str(w_load[i])[1:-1] } }};\n"
            out += f"std::copy(w_loader_l{nb}n{i}, w_loader_l{nb}n{i} + {size[0]}, l{nb}n{i}_w);\n}}\n"

    out += f"\n// Layer {nb} bias\n"
    if nb > 0:
        f = open(f"{path}dense_{nb}_bias_array.csv", 'r')
    else:
        f = open(f"{path}dense_bias_array.csv", 'r')

    b_list = f.read().splitlines()
    f.close()

    for i, b in enumerate(b_list):
        out += f"__PR_L{nb}N{i}__ l{nb}n{i}_b = {b};\n"

    return out


def compute_layer(nb, size, activation_fn):
    if nb == 0:
        input = "INPUT"
    else:
        input = f"l{nb -1}_o"

    out = f"\n__PR_{nb}O__* l{nb}_o = (__PR_{nb}O__*) malloc({size[1]} * sizeof(__PR_{nb}O__));\n"

    if size[0] == 1:
        for i in range(size[1]):
            out += f"l{nb}_o[{i}] = {activation_fn}({input}[{i}] * l{nb}n{i}_w + l{nb}n{i}_b);\n"
    elif size[1] == 1:
        out += f"l{nb}_o[0] = l{nb}n0_b;\n"
        for i in range(size[0]):
            out += f"l{nb}_o[0] += {input}[{i}] * l{nb}n{i}_w;\n"
        out += f"l{nb}_o[0] = {activation_fn}(l{nb}_o[0]);\n"
    else:
        for i in range(size[1]):
            out += f"l{nb}_o[{i}] = l{nb}n{i}_b;\n"
        for i in range(size[1]):
            out += f"for (unsigned i = 0; i < {size[0]}; ++i) l{nb}_o[{i}] += {input}[i] * l{nb}n{i}_w[i];\n"
        if activation_fn == 'softmax':
            out += f"\nsoftmax(l{nb}_o, {size[1]});\n"
        else:
            out += f"\nfor (unsigned i = 0; i < {size[1]}; ++i) l{nb}_o[i] = {activation_fn}(l{nb}_o[i]);\n"
    
    return out


def set_activation(fname):
    if fname == 'tanh':
        return ''
    elif fname == 'relu':
        return "\ndouble relu(double a){ return a < 0. ? (double)0. : a;}\n"
    elif fname == 'softmax':
        out = ''
        for type in ('double', 'float'):
            out += f"static void softmax({type} * input, size_t input_len){{\n" \
                   f"{type} m = -INFINITY;\n" \
                   "for (size_t i = 0; i < input_len; i++) {\n" \
                   "if (input[i] > m) m = input[i];\n}\n" \
                   f"{type} sum = 0.;\n" \
                   "for (size_t i = 0; i < input_len; i++) sum += exp(input[i] - m);\n" \
                   f"const {type} scale = m + log(sum);\n" \
                   "for (size_t i = 0; i < input_len; i++) input[i] = exp(input[i] - scale);\n}\n"
        return out


# if __name__ == 'main':

args = {'name': "sine_", 'layers': ((1, 20), (20, 6), (6, 1)), 'activation': ("tanh", "tanh", "tanh")} # todo: get args via hdf file
f = open('sine_nn.cpp', 'a')
f.write("#include <cmath>\n#include <cstdlib>\n#include <algorithm>\n \n")

for fn in args['activation']:
    f.write(set_activation(fn))  # todo: case with the same activation fn

f.write("\nint main(){\n")
f.write(f"double INPUT = {{{str([.5 for _ in range(args['layers'][0][0])])[1:-1]}}};")  # to change if needed

for i, sz in enumerate(args['layers']):
    f.write(parse_layer(i, args['name'], sz))

f.write("\n//Computing\n")

for i, sz in enumerate(args['layers']):
    f.write(compute_layer(i, sz, args['activation'][i]))

f.write(f"\nPROMISE_CHECK_ARRAY(l{len(args['layers']) - 1}_o, {args['layers'][-1][1]});\n")
f.write("\nreturn EXIT_SUCCESS;\n}\n")
f.close()
