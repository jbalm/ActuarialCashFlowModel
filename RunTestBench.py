#obj = TestStringMethods()
#obj.test_full_module()
from os import path, sys
BASE_DIR = path.dirname(path.dirname(path.dirname(__file__)))  # ands
sys.path.append(BASE_DIR)

if __name__ == "__main__":
    from .code.core_math.test.test_functions_credit import test_functions_credit 
    print(BASE_DIR)
    test_functions_credit.test_full()


    
    
#command = "python .core_math.test.test_functions_credit.py"
#proc = subprocess.Popen(command,stdout=subprocess.PIPE,shell=True)
#(out, err) = proc.communicate()
#outwithoutreturn = out.rstrip('\n')
#print(out)
#execfile("python .core_math.test.test_functions_credit.py")
#a = os.system("python .core_math.test.test_functions_credit.py")
#print(a)
#exec(open("core_math/test/test_functions_credit.py").read())
#
#obj = TestStringMethods()
#        obj.test_fact()