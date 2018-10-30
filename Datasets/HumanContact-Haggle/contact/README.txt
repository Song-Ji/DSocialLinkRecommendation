Haggle network, part of the Koblenz Network Collection
===========================================================================

This directory contains the TSV and related files of the contact network:

This undirected network represents contacts between people measured by carried wireless devices. A node represents a person; an edge between two persons shows that there was a contact between them.


More information about the network is provided here:
http://konect.uni-koblenz.de/networks/contact

Files:
    meta.contact -- Metadata about the network
    out.contact -- The adjacency matrix of the network in space separated values format, with one edge per line
      The meaning of the columns in out.contact are:
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

@MISC{konect:2016:contact,
    title = {Haggle network dataset -- {KONECT}},
    month = sep,
    year = {2016},
    url = {http://konect.uni-koblenz.de/networks/contact}
}

@article{konect:chaintreau07,
        title = {Impact of Human Mobility on Opportunistic Forwarding Algorithms},
        author = {Augustin Chaintreau and Pan Hui and Jon Crowcroft and
                  Christophe Diot and Richard Gass and James Scott},
        journal = {IEEE Trans. on Mobile Computing},
        year = {2007},
        volume = {6},
        number = {6},
        pages = {606-620 },
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
