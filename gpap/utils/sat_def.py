# global file for some SAT definitions

# data type for external lit
ext_lit = int
# data type for external clause
ext_cls = list[ext_lit]

# an external assignment is a list of literals
ext_assg = list[ext_lit]

# multiple assumption sets (each is a list of literals)
ext_assumptions = list[ext_assg]
