from rubik_solver import utils

def main():
    # state variable comes directly from cubenew.py
    ex_state=  {
                'up':['white','white','white','white','white','white','white','white','white',],
                'right':['white','white','white','white','white','white','white','white','white',],
                'front':['white','white','white','white','white','white','white','white','white',],
                'down':['white','white','white','white','white','white','white','white','white',],
                'left':['white','white','white','white','white','white','white','white','white',],
                'back':['white','white','white','white','white','white','white','white','white',]
    }
    solution = solve(ex_state)
    print(solution)

def solve(state):
    versions = ['Beginner', 'CFOP', 'Kociemba'] # ordered slowest to fastest
    version = versions[2]

    # TODO: Turn state into cube string
    # ex
    cube = 'wowgybwyogygybyoggrowbrgywrborwggybrbwororbwborgowryby'
    return utils.solve(cube, version)

if __name__=="__main__":
    main()
    