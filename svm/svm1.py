import sys,math,plot
 
# get current tolerance level
def tolerance(v1,v2,norm):
    sum = 0.0
    for k in v1.keys()+v2.keys():
        t1 = 0
        t2 = 0
        if k in v1:
            t1 = v1[k]
        if k in v2:
            t2 = v2[k]
        sum += (t1-t2) * (t1 - t2)
         
    return math.sqrt(sum/norm)
 
# fill dense alpha list with initial values
def init_alpha(num_examples):
    alpha_s = []
    for i in range(num_examples):
        alpha_s.append((0.0,i))
    return alpha_s
 
# dot products for two dict format vectors
def dot(v1,v2):
    sum = 0.0
    if len(v1) < len(v2):
        for id in v1.keys():
            if id in v2:
                sum += v1[id] * v2[id]
    else:
        for id in v2.keys():
            if id in v1:
                sum += v1[id] * v2[id]
    return sum
 
# kernel evaluation for two vectors
def kernel(v1, v2):
    return dot(v1,v2)
 
# build lower triangular kernel matrix compute_Q
def compute_Q(es,height):
    q = []
    for i in range(height):
        q.append([])
        for j in range(height):
            if i>=j:
                q[i].append(kernel(es[i],es[j]))
                 
    return q
 
# retrieve kernel matrix element on i,j 
def ret_Q(q,i,j):
    if i>=j:
        return q[i][j]
    else:
        return q[j][i]
 
def read_examples(stream):
    es = []
    cs = []
    width = 0
    height = 0
     
    for line in stream:
        e = {}
        c = float(line[0:line.find(" ")])
        for t in line[line.find(" ")+1:].split():
            ts = t.split(":")
            id = int(ts[0])
            e[id] = float(ts[1])
            if id > width:
                width = id
             
        es.append(e)
        cs.append(c)
        height += 1
    return height, width, cs,es
 
def main(C=1.0, max_iter=100, tol = 0.001, show_plot=False):
     
    alpha_sparse = {} # current sparse alpha list
    alpha_s_prim = {} # previous sparse alpha list
     
    print >> sys.stderr, "loading examples..."
    height_e, width_e, cs,es = read_examples(sys.stdin)
     
    print >> sys.stderr, "example matrix: " , height_e, ",", width_e
    print >> sys.stderr, "kernelizing..."
    q = compute_Q(es,height_e)
     
    alpha_s = init_alpha(height_e)
    bias = 0
    # stochastic gradient descent
    for i in range(max_iter):
        print >> sys.stderr, "¥nnew iteration:", i+1
        gamma = 1
         
        # sort alpha list in reversed order
        alpha_s.sort(None, None, True)
        print >> sys.stderr, alpha_s[0:30]
        print >> sys.stderr, 'sparsity: ', len(alpha_sparse),':',height_e
         
        alpha_s_prim = alpha_sparse.copy()
         
        z_max = float("-infinity"); z_min= float("infinity")
         
        for id in range(len(alpha_s)):
            # update from the largest alpha
            alpha = alpha_s[id][0]
            j = alpha_s[id][1]
            t = 0.0
             
            for k in alpha_sparse.keys():
                t += cs[k]* alpha_sparse[k] * ret_Q(q,j,k)
            # check z_max and z_min for bias computation
            if cs[j]>0:
                if t < z_min:
                    z_min = t
            else:
                if t > z_max:
                    z_max = t
                     
            learning_rate = gamma * (1/ret_Q(q,j,j))
            delta = learning_rate * ( 1 - t * cs[j] )
             
            # check for soft-margin
            alpha += delta
            if alpha < 0 :
                alpha = 0.0
            if alpha > C:
                alpha = C
             
            # do update foe dense alpha list
            alpha_s[id] = alpha,j
             
            # do update for sparse alpha list
            if math.fabs(alpha - 0.0) >= 1e-10:
                alpha_sparse[j]=alpha
            else:
                if j in alpha_sparse:
                    del alpha_sparse[j]
        # get bias
        bias = (z_max+z_min)/2.0
 
        # chekc for tolerance
        tol1 = tolerance(alpha_sparse, alpha_s_prim, float(height_e))
             
        print >> sys.stderr, "tolerance:", tol1
        if tol1 < tol:
            print >> sys.stderr, "¥nfinished in",i+1,"iterations"
            break
     
    svm_res ={'sv_s':[],'id_s':[],'alpha_s':[]}
    # support vectors
    for id,alpha in alpha_sparse.items():
        svm_res['sv_s'].append(es[id])
        svm_res['id_s'].append(id)
        svm_res['alpha_s'].append(cs[id]*alpha)
    svm_res['bias'] = bias
    # plot graph if needed
    if show_plot:
        plot.draw(cs, es, svm_res)
    return svm_res
 
if __name__ == "__main__":
    t= main()
    print 'support vectors:', t['sv_s']
    print 'example IDs:', t['id_s']
    print 'lagrange multipliers:',t['alpha_s']
    print 'bias:', t['bias']
