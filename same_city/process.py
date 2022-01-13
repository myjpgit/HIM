
def node_index():
    nodes=set()
    with open('e:/data/terrorattack/terrorist_attack.nodes') as file:
        for line in file:
            line=line.strip().split('\t')
            nodes.add(line[0])
    print(len(nodes))
    with open('refined_data/node_ind.txt','w') as file:
        for n in nodes:
            file.write(n+'\n')

def get_node_index():
    node_ind={}
    count=0
    with open('refined_data/node_ind.txt') as file:
        for line in file:
            line=line.strip()
            node_ind[line]=count
            count+=1
    return node_ind

def relation_true():
    node_ind=get_node_index()
    out_file=open('data/relation_true.txt','w')
    with open('e:/data/terrorattack/terrorist_attack_loc_org.edges') as file:
        for line in file:
            line=line.strip().split(' ')
            id1=node_ind[line[0]]
            id2=node_ind[line[1]]
            out_file.write(str(id1)+'\t'+str(id2)+'\n')

def get_true_pair():
    true_pair=set()
    with open('data/relation_true.txt') as file:
        for line in file:
            line=line.strip().split('\t')
            id1=int(line[0])
            id2=int(line[1])
            true_pair.add((id1,id2))
            true_pair.add((id2,id1))
    return true_pair

def relation_false():
    true_pair=get_true_pair()
    node_ind=get_node_index()
    out_file = open('data/relation_false.txt', 'w')
    with open('e:/data/terrorattack/terrorist_attack_loc.edges') as file:
        for line in file:
            line=line.strip().split(' ')
            id1=node_ind[line[0]]
            id2=node_ind[line[1]]
            if not (id1,id2) in true_pair:
                out_file.write(str(id1) + '\t' + str(id2) + '\n')



if __name__=='__main__':
    # node_index()
    # relation_true()
    relation_false()