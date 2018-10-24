Wikipedia elections network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the elec network:

This is the network of users from the English Wikipedia that voted for and against each other in admin elections. Nodes represent individual users, and edges represent votes. Edges can be positive ("for" vote) and negative ("against" vote). Each edge is annotated with the date of the vote. In the original dataset from the SNAP website, certain timestamps are from after 2050; the corresponding edges are not included in this version of the network.


More information about the network is provided here: 
http://konect.uni-koblenz.de/networks/elec

Files: 
    meta.elec -- Metadata about the network 
    out.elec -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.elec are: 
        First column: ID of from node 
        Second column: ID of to node
        Third column: edge weight
        Fourth column: timestamp of the edge


Complete documentation about the file format can be found in the KONECT
handbook, in the section File Formats, available at:

http://konect.uni-koblenz.de/publications

All files are licensed under a Creative Commons Attribution-ShareAlike 2.0 Germany License.
For more information concerning license visit http://konect.uni-koblenz.de/license.



Use the following References for citation:

@MISC{konect:2016:elec,
    title = {Wikipedia elections network dataset -- {KONECT}},
    month = sep,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/elec}
}

@inproceedings{konect:leskovec207,
        title = {Governance in Social Media: A Case Study of the
                  {Wikipedia} Promotion Process},  
        author = {Jure Leskovec and Daniel Huttenlocher and Jon
                  Kleinberg},
        booktitle = {Proc. Int. Conf. on Weblogs and Social Media},
        year = {2010},
}


@inproceedings{konect,
	title = {{KONECT} -- {The} {Koblenz} {Network} {Collection}},
	author = {Jérôme Kunegis},
	year = {2013},
	booktitle = {Proc. Int. Conf. on World Wide Web Companion},
	pages = {1343--1350},
	url = {http://userpages.uni-koblenz.de/~kunegis/paper/kunegis-koblenz-network-collection.pdf}, 
	url_presentation = {http://userpages.uni-koblenz.de/~kunegis/paper/kunegis-koblenz-network-collection.presentation.pdf},
}


