Prosper loans network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the prosper-loans network:

These are loans between users of the Prosper.com website.  The network is directed and denotes who loaned money to whom. 


More information about the network is provided here: 
http://konect.uni-koblenz.de/networks/prosper-loans

Files: 
    meta.prosper-loans -- Metadata about the network 
    out.prosper-loans -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.prosper-loans are: 
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

@MISC{konect:2016:prosper-loans,
    title = {Prosper loans network dataset -- {KONECT}},
    month = oct,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/prosper-loans}
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


