#!/usr/bin/env python
import os, re, sys
with open(sys.argv[1], 'r') as f:
    supp_text = f.read()
known_labels = []
label_pat = re.compile(r'\\label{([^}]*)}')
matches = label_pat.finditer(supp_text)
for match in matches:
    #known_labels.extend([label.strip() for label in match.group(1).split(',')])
    known_labels.append(match.group(1).strip())
#print(f'All labels:\n{known_labels}')
ref_commands_base = ['Cite', 'cite', 'ref']
ref_pats = [re.compile(r'\\.*' + base + r'(\[[^\]]*\])?{([^}]*)}') for base in ref_commands_base]
new_supp_text = supp_text
lines = supp_text.splitlines(True)
line_lengths = [len(line) for line in lines]
cumulative_lengths = [sum(line_lengths[:i+1]) for i in range(len(lines))]
print(cumulative_lengths)
for pat in ref_pats:
    current_text = ''
    current_pos = 0
    matches = pat.finditer(new_supp_text)
    for match in matches:
        all_text = match.group()
        if 'hyperref' in all_text:
            labels_with_brackets, text = match.group(1,2)
            labels = label_with_brackets[1:-1]
        else:
            options, labels = match.group(1,2)
        labels = labels.split(',')

        start, end = match.span()
        line_no = next((no+1 for no, length in enumerate(cumulative_lengths) if length >= start), len(lines)+1)
        #print(repr(new_supp_text[end-1:end+2]))
        #print('C{:<7} {:<45}'.format(str(start)+':', all_text), end='')
        print(f'L{line_no:<6} {all_text:<45}', end='')
        label_not_found = False
        for label in labels:
            if label not in known_labels:
                label_not_found = True
                break
        if label_not_found:
            print(f'label {label} NOT found, wrapping with \\nolink{{}}')
            replacement = r'\nolink{'+all_text+r'}'
            current_text += new_supp_text[current_pos:start] + replacement
        else:
            print('labels found')
            current_text += new_supp_text[current_pos:end]
        current_pos = end
    if current_pos < len(new_supp_text):
        current_text += new_supp_text[current_pos:]
    new_supp_text = current_text
with open('out.tex', 'w') as f:
    f.write(new_supp_text)

# TESTING using https://regex101.com/ (Flavour: Python)
# \\.*ref(\[[^\]]*\])?{([^}]*)}
#\ref{eq:1}
#\eqref{eq:1}
#\eqref[]{eq:1}
#\hyperref[label]{text}
