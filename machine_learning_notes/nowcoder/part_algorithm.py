
#em algorithm
def calcEM(height):
    N = len(height)
    gp = 0.5
    bp = 0.5
    gmu,gsigma = min(height),1
    bmu,bsigma = max(height),1
    ggamma = range(N)
    bgamma = range(N)
    cur = [gp, bp, gmu,gsigma,bmu,bsigma]
    now = []

    times = 0
    while times < 100:
    	i =0
        for x in height:
            ggamma[i] = gp * gauss(x,gmu,gsigma)
            bgamma[i] = bp * gauss(x,bmu,bsigma)
            s = ggamma[i] + bgamma[i]
            ggamma[i] /= s
            bgamma[i] /= s
            i += 1
        gn = sum(ggamma)
        gp = float(gn) / float(N)
        bn = sum(bgamma)
        bp = float(bn) / float(N)
        gmu = averageWeight(height, ggamma, gn)
        gsigma = varianceWeight(height,ggamma,gmu,gn)
        bmu = averageWeight(height,bgamma,bn)
        bsigma = varianceWeight(height, bgamma, bmu,bn)

        now = [gp, bp, gmu, gsigma, bmu, bsigma]
        if isSame(cur, now):
            break
        cur = now
        print "Times:\t",times
        print "Girl mean/gsigma:\t",gmu,gsigma
        print "Boy mean/bsigma:\t",bmu,bsigma
        print "Boy/Girl:\t",bn,gn,bn+gn
        print "\n\n"
        times += 1
    return now

#gmm algorithm
if __name__ == "__main__":
    im = Image.open('.\\Pic\\test.bmp')	
    print im.format,im.size,im.mode
    
    im = im.split()[0]
    nb = []
    data = list(im.getdata())
    parameter = GMM(data)
    t = composite(data, parameter)

    im1 = Image.new('L', im.size)
    im1.putdata(t[0])

def composite(band, parameter):
    c1 = parameter[0]
    mu1 = parameter[2]
    sigma1 = parameter[3]
    c2 = parameter[1]
    mu2 = parameter[4]
    sigma2 = parameter[5]

    p1 = []
    p2 = []
    for pixel in band:
        p1.append(c1 * gauss(pixel, mu1, sigma1))
        p2.append(c2 * gauss(pixel, mu2,sigma2))
    scale(p1)
    scale(p2)
    return [p1,p2]    


#谱聚类 algorithm
def spectral_cluster(data):
    lm = laplace_matrix(data)
    eg_values,eg_vectors = linalg.eig(lm)
    idx = eg_values.argsort()
    eg_vectors = eg_vectors[:,idx]

    m = len(data)
    eg_data = [[] fro x in range(m)]
    for i in range(m):
        eg_data[i] = [0 for x in range(k)]
        for j in range(k):
            eg_data[i][j] = eg_vectors[i][j]
    return k_means(eg_data)
def laplace_matrix(data):
    m = len(data)
    w = [[] for x in range(m)]
    for i in range(m):
        w[i] = [0 for x in range(m)]
    nearest = [0 for x in range(neighbor)]    
    for i in range(m):
        zero_list(nearest)
        for j in range(i+1, m):
            w[i][j] = similar(data,i,j)
            if not is_neighbor(w[i][j],nearest):
                w[i][j]=0
            w[j][i] = w[i][j] #对称
        w[i][i] = 0
    for i in range(m):
        s = 0
        for j in range(m):
            s+= w[i][j]
        if s == 0:
            print "矩阵第",i, "行全为0"
            continue
        for j in range(m):
            w[i][j] /= s
            w[i][j] = -w[i][j]
        w[i][i] += 1  #单位阵主对角线为1


#svm gauss_kernel function
def update(i,j,data):
    low = 0
    high = C
    if data[i][-1] == data[j][-1]:
        low = max(0,alpha[i] + alpha[j] - C)	
        high = min(C, alpha[i] + alpha[j])
    else:
        low = max(0,alpha[j] - alpha[i])
        high = min(C, alpha[j] - alpha[i] + C)
    if low == high:
        return False
    eta = kernel(data[i], data[i]) + kernel(data[j],data[j]) - 2* kernel(data[i],data[j])
    if is_same(eta, 0):
        return False
    ei = predict(data[i], data) - data[i][-1]
    ej = predict(data[j], data) - data[j][-1]
    alpha_j = alpha[j] + data[j][-1] * (ei - ej) / eta
    if alpha_j == alpha[j]:
        return False
    if alpha_j > high:
        alpha_j = high
    elif alpha_j < low:
        alpha_j = low
    alpha[i] += (alpha[j] - alpha_j) * data[i][-1] * data[j][-1]
    alpha[j] = alpha_j
    return True

#decision_tree
def decision_tree():
    m = len(data)
    n = len(data[0])
    tree = []
    root = TreeNode()
    if rf:
        root.sample = random_select(alpha)
    else:
        root.sample = [x for in range(m)]
    root.gini_coefficient()
    tree.append(root)
    first = 0
    last = 1
    for level in range(max_level):
        for node in range(first,last):
            tree[node].split(tree)
        first = last
        last = len(tree)
        print level+1,len(tree)
    return tree
#use gradient_descent to get theta
def calCoefficient(data,listA,listW, listLostFunction):
    N = len(data[0]) #dimenson
    w = [0 for i in range(N)]
    wNew = [0 for i in range(N)]
    g = [0 for i in range(N)]

    times = 0
    alpha = 100.0  #learn_rate or step_size
    while times < 10000:
        j = 0
        while j < N:
            g[j] = gradient(data, w, j)
            j += 1
        normalize(g) #regular_gradient
        alpha = calcAlpha(w, g, alpha, data)    
        numberProduct(alpha, g, wNew)

        print "times,alpha,fw,w,g:\t",times, alpha, fw(w,data),w,g
        if isSame(w, wNew):
            break
        assign2(w,wNew)
        times += 1

        listA.append(alpha) # list in order to draw a picture
        listW.append(assign(w))
        listLostFunction.append(fw(w,data))
    return w    

#use A criteria to get learn rate(step_size)
#w: current value
#g: current gradient direction
#a: current learn_rate
def calAlpha(w,g,a,data):
    c1 = 0.3
    now = fw(w,data)
    wNext = assign(w)
    numberProduct(a,g,wNext)
    next = fw(wNext, data)
    # looking for a large enough,so that h(a)>0
    count = 30
    while next < now:
        a *= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data)
        count -= 1
        if count == 0
            break
    #looking for suitable learn_rate -> a
    count = 50
    while next > now - c1 * a * dotProduct(g,g):
        a /= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data)

        count -= 1
        if count == 0:
            break
    return a
