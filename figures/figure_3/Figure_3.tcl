if {1} {
update idletasks
namespace eval ::Figure3 {
    set state_path [info script]
    if {$state_path eq ""} {
        set process_args [file join [file separator] proc [pid] cmdline]
        if {[file readable $process_args]} {
            set process_handle [open $process_args r]
            set process_words [split [read $process_handle] \x00]
            close $process_handle
            set e_index [lsearch -exact $process_words -e]
            if {$e_index >= 0} {set state_path [lindex $process_words [expr {$e_index + 1}]]}
        }
    }
    if {$state_path ne ""} {
        variable base [file dirname [file normalize $state_path]]
    } else {
        variable base [pwd]
    }
    if {$state_path eq "" && ![file isfile [file join $base data reference protein_reference.pdb]] && [file isfile [file join $base figure_3 data reference protein_reference.pdb]]} {
        set base [file join $base figure_3]
    }
    if {![file isfile [file join $base data reference protein_reference.pdb]]} {
        error "Cannot locate this figure's data folder. Change into the figure folder and run vmd -e Figure_3.tcl, or load the state with File > Load Visualization State."
    }
    variable ready 0
    variable molecule_ids {}
    variable protein_id
    proc load_molecule {relative type title} {
        variable base
        variable molecule_ids
        set path [file join $base $relative]
        if {![file isfile $path]} {error "Missing structure: $relative"}
        set m [mol new $path type $type waitfor all autobonds off]
        if {[molinfo $m get numframes] == 0} {error "No coordinate frame: $relative"}
        mol rename $m $title
        if {$type eq "mol2"} {
            set all_atoms [atomselect $m all]
            set elements [$all_atoms get type]
            $all_atoms set element $elements
            set mass_table [dict create C 12.011 N 14.007 O 15.999 S 32.06 F 18.998 H 1.008 Cl 35.45 Br 79.904 P 30.974]
            set radius_table [dict create C 1.70 N 1.55 O 1.52 S 1.80 F 1.47 H 1.20 Cl 1.75 Br 1.85 P 1.80]
            set masses {}; set radii {}
            foreach element $elements {
                lappend masses [dict get $mass_table $element]
                lappend radii [dict get $radius_table $element]
            }
            $all_atoms set mass $masses
            $all_atoms set radius $radii
            $all_atoms delete
        }
        mol delrep 0 $m
        lappend molecule_ids $m
        return $m
    }
    proc add_rep {m selection style color material_name} {
        set sel [atomselect $m $selection]
        if {[$sel num] == 0} {$sel delete; error "Empty selection: $selection"}
        $sel delete
        mol selection $selection
        mol representation {*}$style
        mol color ColorID $color
        mol material $material_name
        mol addrep $m
    }
    color change rgb 29 0.20000000298 0.600000023842 0.20000000298
    color change rgb 30 1.0 0.0 1.0
    color change rgb 31 0.808080792427 0.808080792427 0.808080792427
    foreach mat_name {PaperOverlayProtein PaperOverlayLigands} {
        if {[lsearch -exact [material list] $mat_name] < 0} {material add $mat_name}
    }
    material change ambient PaperOverlayProtein 0.35
    material change diffuse PaperOverlayProtein 0.65
    material change specular PaperOverlayProtein 0.05
    material change opacity PaperOverlayProtein 0.08
    material change shininess PaperOverlayProtein 0.25
    material change ambient PaperOverlayLigands 0.20
    material change diffuse PaperOverlayLigands 0.70
    material change specular PaperOverlayLigands 0.35
    material change shininess PaperOverlayLigands 0.50
    material change opacity PaperOverlayLigands 1.0
    set protein_id [load_molecule data/reference/protein_reference.pdb pdb {5LQF receptor (NewCartoon)}]
    mol bondsrecalc $protein_id
    mol reanalyze $protein_id
    source [file join $base data protein_secondary_structure.tcl]
    add_rep $protein_id protein {NewCartoon 0.30 20.0 4.1 0} 31 PaperOverlayProtein
    set native_id [load_molecule data/vmd_ligands/native.mol2 mol2 {Native ligand - forest green}]
    add_rep $native_id {not hydrogen} {Licorice 0.19 24.0 24.0} 29 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_01.mol2 mol2 {Candidate 01 - anchor-corrected-20261001-v1-0342-068}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_02.mol2 mol2 {Candidate 02 - anchor-corrected-20261001-v1-0343-047}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_03.mol2 mol2 {Candidate 03 - anchor-corrected-20261001-v1-0344-000}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_04.mol2 mol2 {Candidate 04 - anchor-corrected-20261001-v1-0357-013}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_05.mol2 mol2 {Candidate 05 - anchor-corrected-20261001-v1-0346-052}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    set candidate_id [load_molecule data/vmd_ligands/candidate_06.mol2 mol2 {Candidate 06 - anchor-corrected-20261001-v1-0347-101}]
    add_rep $candidate_id {not hydrogen} {Licorice 0.105 24.0 24.0} 30 PaperOverlayLigands
    display projection Orthographic
    display height 6.0
    display distance -2.0
    display focallength 2.0
    display resize 1200 900
    display nearclip set 0.01
    display farclip set 100.0
    display depthcue off
    display shadows off
    display ambientocclusion off
    display backgroundgradient off
    color Display Background white
    axes location Off
    stage location Off
    light 0 on
    light 1 on
    light 2 off
    light 3 off
    set center_matrix {{1 0 0 0.401757717133} {0 1 0 -16.2879581451} {0 0 1 -42.1751251221} {0 0 0 1}}
    set rotation_matrix {{0.522354960442 0.582883059978 -0.622408747673 0} {-0.220909312367 0.79748415947 0.56144285202 0} {0.823616683483 -0.155776560307 0.545334100723 0} {0 0 0 1}}
    set scale_matrix {{0.139896104293 0 0 0} {0 0.139896104293 0 0} {0 0 0.139896104293 0} {0 0 0 1}}
    set identity {{1 0 0 0} {0 1 0 0} {0 0 1 0} {0 0 0 1}}
    foreach m $molecule_ids {
        molinfo $m set center_matrix [list $center_matrix]
        molinfo $m set rotate_matrix [list $rotation_matrix]
        molinfo $m set scale_matrix [list $scale_matrix]
        molinfo $m set global_matrix [list $identity]
        mol on $m
    }
    foreach m $molecule_ids {
        mol clipplane center 0 0 $m {10.1954641371 14.2836291868 49.1917708092}
        mol clipplane normal 0 0 $m {-0.823616683483 0.155776560307 -0.545334100723}
        mol clipplane status 0 0 $m 1
        mol clipplane center 1 0 $m {-10.9989795714 18.2922871035 35.1584794349}
        mol clipplane normal 1 0 $m {0.823616683483 -0.155776560307 0.545334100723}
        mol clipplane status 1 0 $m 1
    }
    variable ready 1
    mol top $protein_id
    display update
    puts "Loaded receptor, native ligand, and six candidates. Graphics > Representations edits each object."
    puts "The receptor uses the paper's faint cartoon (opacity 0.08); choose Opaque material for stronger contrast."
}
}
