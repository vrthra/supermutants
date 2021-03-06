import csv, sys

class O:
    def log(*v):
        print(*v)

# Find functions that are reachable from fnA
def find_reachable(call_g, fnA, reachable_keys=None, found_so_far=None):
    if reachable_keys is None: reachable_keys = {}
    if found_so_far is None: found_so_far = set()

    for fnB in call_g[fnA]:
        if fnB in found_so_far: continue
        found_so_far.add(fnB)
        if fnB not in call_g: continue
        if fnB in reachable_keys:
            for k in reachable_keys[fnB]:
                found_so_far.add(k)
        else:
            keys = find_reachable(call_g, fnB, reachable_keys, found_so_far)
    return found_so_far

# Produce reachability dictionary given the call graph.
# For each function, we have functions that are reachable from it.
def reachable_dict(call_g):
    reachable = {}
    for fnA in call_g:
        keys = find_reachable(call_g, fnA, reachable)
        reachable[fnA] = keys
    return reachable

# Given the call graph, produce the reachability matrix with 1 for reachability
# and 0 for non reachability. The rows and columns are locations
def reachability_matrix(call_g):
    reachability = reachable_dict(call_g)
    rows = list(reachability.keys())
    columns = list(reachability.keys())
    matrix = []
    for i in rows:
        matrix_row = []
        for j in columns:
            matrix_row.append(1 if i in reachability[j] or j in reachability[i] or i == j else 0)
            #matrix_row.append(1 if i in reachability[j] or i == j else 0)
        matrix.append(matrix_row)
    return matrix, columns


# Insert the location information to matrix
def insert_names(matrix, keys):
    new_m = []
    for i,row in enumerate(matrix):
        new_r = []
        for j,c in enumerate(row):
            new_r.append((keys[j], c))
        new_m.append((keys[i], new_r))
    return new_m

# Remove a single row corresponding to a location from matrix
def remove_single_row(named_matrix, location):
    return [(k, row) for k,row in named_matrix if k != location]
 
# Remove a single column corresponding to a location from matrix
def remove_single_column(named_matrix, location):
    return [(k, [(k1,c) for k1,c in row if k1 != location]) for k,row in named_matrix]

# Remove a row and column corresponding to location
def remove_row_column(named_matrix, loc):
    m1 = remove_single_row(named_matrix, loc)
    m2 = remove_single_column(m1, loc)
    return m2

import random
def get_chosen_row(named_matrix, mutants):
    remaining_loc = [(loc,row) for loc,row in named_matrix if mutants[loc]]
    if remaining_loc:
        loc,row = random.choice(remaining_loc)
        return loc,row
    return None, None

def identify_superlocations(matrix, locations, mutants):
    named_matrix = insert_names(matrix, locations)
    my_locations = []
    while named_matrix:
        loc, row = get_chosen_row(named_matrix, mutants)
        if loc is None: break
        my_locations.append(loc)

        m1 = remove_row_column(named_matrix, loc)
        named_matrix = m1

        # now find locations which are 1
        reachable = [k1 for k1,c in row if c == 1]
        for loc_ in reachable:
            m1 = remove_row_column(named_matrix, loc_)
            named_matrix = m1

    return my_locations

def total_mutants(mutants):
    m =  sum([len(mutants[loc]) for loc in mutants])
    return m

def generate_supermutants(matrix, mutants):
    all_supermutants = []
    while total_mutants(mutants):
        superloc = identify_superlocations(matrix, keys, mutants)
        assert superloc
        # now choose one mutant per location
        my_mutants = []
        for loc in superloc:
            m,*rest = mutants[loc]
            mutants[loc] = rest
            my_mutants.append((loc, m))
        all_supermutants.append(my_mutants)
        O.log('remaining mutants:', total_mutants(mutants), '/', TOTAL_MUTANTS)
        O.log('supermutants:', len(all_supermutants))
    return all_supermutants

def print_matrix(matrix, keys):
    print('\t' + ',\t'.join(keys))
    for i,row in enumerate(matrix):
        print(keys[i] + '\t' + ',\t'.join([str(c) for c in row]))

def load_mutants(fname, locations):
    my_m = {}
    unknown_functions = []
    with open(fname) as f:
        lines = csv.DictReader(f)
        for line in lines:
            mID, fn = line['mutID'], line['fnName']
            if fn not in locations:
                unknown_functions.append(fn)
                continue
            if fn not in  my_m: my_m[fn] = []
            my_m[fn].append(mID)
    for loc in locations:
        if loc not in my_m:
            my_m[loc] = []
    return my_m, unknown_functions

def load_call_graph(fname):
    my_g = {}
    called = {}
    with open(fname) as f:
        lines = csv.DictReader(f)
        for line in lines:
            fnA, fnB = line['Caller'], line['Callee']
            if fnA not in  my_g: my_g[fnA] = []
            my_g[fnA].append(fnB)
            called[fnB] = True
    # now, populate leaf functions
    for b in called:
        if b not in my_g: my_g[b] = []
    return my_g

CALL_GRAPH = load_call_graph(sys.argv[1])
MUTANTS, unknown = load_mutants(sys.argv[2], CALL_GRAPH.keys())
O.log('Unknown:', unknown)
TOTAL_MUTANTS = total_mutants(MUTANTS)

matrix, keys = reachability_matrix(CALL_GRAPH)
O.log('computed reachability')
# print_matrix(matrix, keys)
supermutants = generate_supermutants(matrix, MUTANTS)
O.log('generated supermutants')
for m in supermutants:
    print(m)
