path=.
pychimera -c 'import chimera'

#for dataset in general-set-except-refined refined-set; do
for dataset in coreset; do
    echo $dataset
    for pdbfile in $path/$dataset/*/*_pocket.pdb; do
        mol2file=${pdbfile%pdb}mol2
        rm $mol2file
	if [[ ! -e $mol2file ]]; then
            echo $mol2file
            #echo -e "open $pdbfile \n addh \n addcharge \n write format mol2 0 tmp.mol2 \n stop" | chimera --nogui
            # Do not use TIP3P atom types, pybel cannot read them
            pychimera script.py $pdbfile
            sed 's/H\.t3p/H    /' tmp.mol2 | sed 's/O\.t3p/O\.3  /' > $mol2file
        fi
    done 
done
