def split(md_table):
    rows = md_table.strip().split("\n")
    header = rows[0].split("|")[1:-1]
    data_rows = [row.split("|")[1:-1] for row in rows[1:]]
    return header, data_rows


def markdown_to_latex_single(markdown_table):
    header, data_rows = split(markdown_table)
    latex_code = ""
    for row in data_rows:
        tau, beta, *values = [cell.strip() for cell in row]

        beta = beta.replace("$", "")
        if beta == "0.5 \mu":
            beta = "$\\frac{1}{2} \mu$"
        else:
            beta = f"${beta}$"

        latex_code += f"& {beta}"
        for value in values:
            mean, std = [s.strip() for s in value.split("±")]
            mean = mean.replace("%", "")
            # cast mean to one decimal
            mean = f"{float(mean):.1f}"
            std = std.replace("%", "")
            # cast std to one decimal
            std = f"{float(std):.1f}"
            latex_code += f" & ${mean}\\% \\pm {std}\\%$"

        latex_code += " \\\\\n"
    return latex_code


def markdown_to_latex(markdown_table_greedy, markdown_table_louvain):
    header_g, data_rows_g = split(markdown_table_greedy)
    header_l, data_rows_l = split(markdown_table_louvain)

    latex_code = ""

    for row_g, row_l in zip(data_rows_g, data_rows_l):
        tau_g, beta_g, *values_g = [cell.strip() for cell in row_g]
        tau_l, beta_l, *values_l = [cell.strip() for cell in row_l]

        beta_g = beta_g.replace("$", "")
        if beta_g == "0.5 \mu":
            beta_g = "$\\frac{1}{2} \mu$"
        else:
            beta_g = f"${beta_g}$"

        latex_code += f"& {beta_g}"
        for value_g in values_g:
            mean_g, std_g = [s.strip() for s in value_g.split("±")]
            mean_g = mean_g.replace("%", "")
            std_g = std_g.replace("%", "")
            latex_code += f" & ${mean_g}\\% \\pm {std_g}\\%$"

        # Process row_l
        beta_l = f"${beta_l}$"
        # latex_code += f" & {beta_l}"
        for value_l in values_l:
            mean_l, std_l = [s.strip() for s in value_l.split("±")]
            mean_l = mean_l.replace("%", "")
            std_l = std_l.replace("%", "")
            latex_code += f" & ${mean_l}\\% \\pm {std_l}\\%$"

        latex_code += " \\\\\n"

    return latex_code


# Inserisci le tabelle Markdown
markdown_table_greedy = """
|   τ | β         | DRL-Agent (ours)   | Random         | Degree         | Centrality     | Roam           |
| 0.5 | 0.5 $\mu$ | 31.67% ± 5.26%     | 20.33% ± 4.55% | 10.33% ± 3.44% | 12.00% ± 3.68% | 12.00% ± 3.68% |
| 0.5 | 1 $\mu$   | 32.33% ± 5.29%     | 27.00% ± 5.02% | 13.00% ± 3.81% | 19.00% ± 4.44% | 12.33% ± 3.72% |
| 0.5 | 2 $\mu$   | 36.00% ± 5.43%     | 33.67% ± 5.35% | 16.33% ± 4.18% | 22.33% ± 4.71% | 18.33% ± 4.38% |
"""

markdown_table_louvain = """
|   τ |   β | DRL-Agent (ours)   | Safeness    | Modularity   |
| 0.3 |   1 | 0.51 ± 0.02        | 0.54 ± 0.07 | 0.49 ± 0.13  |
| 0.3 |   3 | 0.50 ± 0.02        | 0.48 ± 0.07 | 0.54 ± 0.04  |
| 0.3 |   5 | 0.51 ± 0.02        | 0.54 ± 0.10 | 0.59 ± 0.08  |
"""

# Converti le tabelle Markdown in codice LaTeX
# latex_code_greedy = markdown_to_latex(markdown_table_greedy, markdown_table_louvain)
latex_code_greedy = markdown_to_latex_single(markdown_table_greedy)

# Stampa il codice LaTeX
print(latex_code_greedy)
