import os

def modify_activate(file: str, old_str: str, new_str: str):
    """
    usage: modify_activate("/root/.local/share/virtualenvs/autogbm-sgajus1C/bin/activate", "((autogbm) )", "(autogbm-sgajus1C)")
    note: only one {old_str} in the whole file, then replace; otherwise, do nothing.
    """
    with open(file, "r", encoding="utf-8") as f1:
        count = 0
        for line in f1:
            if old_str in line:
                count += 1

    if count != 1:
        print("\033[01;31;01mWarning: not only one '{}' in the whole file (activate) !!\033[01;00;00m ".format(old_str))
    else:
        with open(file, "r", encoding="utf-8") as f1, open("%s.bak" % file, "w", encoding="utf-8") as f2:
            for line in f1:
                if old_str in line:
                    line = line.replace(old_str, new_str)
                f2.write(line)
    
    os.remove(file)
    os.rename("%s.bak" % file, file)


if __name__ == "__main__":

    print("Runing pipenv_patch.py ...")
    base = "/root/.local/share/virtualenvs/"
    proj_dir = os.path.abspath(".").split("/")[-1]

    base_dirlist = os.listdir(base)
    target = False
    for name in base_dirlist:
        if name[:-9] == proj_dir:
            target = base + name + "/bin/activate"
            print("Target file (activate) location: {}".format(target))
            venvdir_name = name
            break

    if not target:
        print("\033[01;31;01mWarning: not found target file (activate) !!\033[01;00;00m ")

    modify_activate(target, "(({}) )".format(proj_dir), "({})".format(venvdir_name))
    print("Done.")

