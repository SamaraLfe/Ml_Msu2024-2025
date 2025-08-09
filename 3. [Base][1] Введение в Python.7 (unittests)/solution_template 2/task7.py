def find_modified_max_argmax(L,f):
 t=[f(x)for x in L if type(x)==int]
 return(max(t),t.index(max(t)))if t else()