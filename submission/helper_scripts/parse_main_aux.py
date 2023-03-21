#!/usr/bin/env python
import os, re, sys
keep_strs = [r'\\citation', r'\\newlabel', r'\\bibcite']
keep_pats = [re.compile(keep_str) for keep_str in keep_strs]
stop_pat = re.compile(r'\\newlabel{LastBibItem}')
with open(sys.argv[1], 'r') as f:
    main_aux_lines = f.readlines()
label_aux_lines = []
for line in main_aux_lines:
    if stop_pat.match(line) is not None: 
        # N.B. will cut off LastBibItem@cref
        label_aux_lines.append(line)
        break
    for pat in keep_pats:
        if pat.match(line) is not None:
            label_aux_lines.append(line)
            break
with open('out.aux', 'w') as f:
    f.writelines(label_aux_lines)
