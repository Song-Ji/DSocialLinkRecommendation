Reality Mining network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the mit network:

This undirected network contains human contact data among 100 students of the Massachusetts Institute of Technology (MIT), collected by the Reality Mining experiment performed in 2004 as part of the Reality Commons project. The data was collected over 9 months using 100 mobile phones. A node represents a person; an edge indicates that the corresponding nodes had physical contact.


More information about the network is provided here: 
http://konect.uni-koblenz.de/networks/mit

Files: 
    meta.mit -- Metadata about the network 
    out.mit -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.mit are: 
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

@MISC{konect:2016:mit,
    title = {Reality Mining network dataset -- {KONECT}},
    month = sep,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/mit}
}

@article{konect:eagle06,
        title = {{Reality} {Mining}: Sensing Complex Social Systems}, 
        author = {Eagle, Nathan and (Sandy) Pentland, Alex},
        journal = {Personal Ubiquitous Computing},
        volume = {10},
        number = {4},
        year = {2006},
        pages = {255--268},
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


