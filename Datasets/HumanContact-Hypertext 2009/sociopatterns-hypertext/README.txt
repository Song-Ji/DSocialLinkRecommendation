Hypertext 2009 network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the sociopatterns-hypertext network:

This is the network of face-to-face contacts of the attendees of the ACM Hypertext 2009 conference. The ACM Conference on Hypertext and Hypermedia 2009 (HT 2009, http://www.ht2009.org/) was held in Turin, Italy over three days from June 29 to July 1, 2009. In the network, a node represents a conference visitor, and an edge represents a face-to-face contact that was active for at least 20 seconds. Multiple edges denote multiple contacts. Each edge is annotated with the time at which the contact took place. 


More information about the network is provided here: 
http://konect.uni-koblenz.de/networks/sociopatterns-hypertext

Files: 
    meta.sociopatterns-hypertext -- Metadata about the network 
    out.sociopatterns-hypertext -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.sociopatterns-hypertext are: 
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

@MISC{konect:2016:sociopatterns-hypertext,
    title = {Hypertext 2009 network dataset -- {KONECT}},
    month = sep,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/sociopatterns-hypertext}
}

@article{konect:sociopatterns,
	author = {Lorenzo Isella and Juliette Stehlé and Alain Barrat and Ciro
                  Cattuto and Jean-François Pinton and Wouter Van den
                  Broeck}, 
	title = {What's in a Crowd? Analysis of Face-to-Face Behavioral Networks},
	journal = {J. of Theoretical Biology},
	volume = {271},
	number = {1},
	pages = {166--180},
	year = {2011},
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


