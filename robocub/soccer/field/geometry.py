"""Backend-neutral dimensions and non-colliding line geometry, in SI units."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[3]

@dataclass(frozen=True)
class Field:
    id: str = 'medium_14x9'
    length: float = 14.
    width: float = 9.
    goal_width: float = 2.5
    goal_height: float = 1.7
    goal_depth: float = 1.
    post_width: float = .08
    goal_area_depth: float = 1.
    goal_area_width: float = 4.
    penalty_area_depth: float = 3.
    penalty_area_width: float = 6.
    penalty_mark_distance: float = 2.
    center_circle_diameter: float = 3.
    corner_radius: float = .5
    line_width: float = .05
    mark_size: float = .1
    border: float = 1.
    ball_radius: float = .69 / (2 * math.pi)
    ball_mass: float = .43

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for k,v in asdict(self).items() if k != 'id'):
            raise ValueError('All dimensions must be finite and positive')
        if not (self.goal_width < self.goal_area_width < self.penalty_area_width < self.width):
            raise ValueError('Goal/area widths must nest within pitch')
        if not (self.goal_area_depth < self.penalty_area_depth < self.length/2):
            raise ValueError('Goal and penalty depths must nest')
        if self.line_width >= self.corner_radius or self.center_circle_diameter >= self.width:
            raise ValueError('Invalid marking dimensions')

    @property
    def spacing(self):
        return max(self.length + 2*self.goal_depth, self.width) + 2*self.border + 3.

    def legal(self, x, y, radius=.4):
        return math.isfinite(x) and math.isfinite(y) and abs(x) <= self.length/2-radius and abs(y) <= self.width/2-radius

    @classmethod
    def load(cls, profile='medium_14x9'):
        path = ROOT/'App/Configs/field_profiles'/f'{profile}.yaml'
        return cls(**yaml.safe_load(path.read_text()))


def geometry(f: Field):
    """Boxes use full extents. Lines lie inside their area's outer boundary."""
    shapes=[]
    def box(name,pos,size,color,collision=True,yaw=0.):
        shapes.append(dict(name=name,pos=list(pos),size=list(size),color=list(color),collision=collision,yaw=yaw))
    green=(.09,.32,.12,1);white=(.95,.95,.95,1)
    box('turf',(0,0,-.05),(f.length+2*f.border+2*f.goal_depth,f.width+2*f.border,.1),green)
    def segment(name,a,b,width=None):
        w=width or f.line_width
        dx,dy=b[0]-a[0],b[1]-a[1]
        box(name,((a[0]+b[0])/2,(a[1]+b[1])/2,.0015),(math.hypot(dx,dy),w,.002),white,False,math.atan2(dy,dx))
    L,W,w=f.length/2,f.width/2,f.line_width
    for sign in [-1,1]:
        segment(f'touch_{sign}',(-L,sign*(W-w/2)),(L,sign*(W-w/2)))
        segment(f'end_{sign}',(sign*(L-w/2),-W),(sign*(L-w/2),W))
        for label,depth,width in [('goal_area',f.goal_area_depth,f.goal_area_width),('penalty_area',f.penalty_area_depth,f.penalty_area_width)]:
            x=sign*(L-depth+w/2);y=width/2-w/2
            segment(f'{label}_front_{sign}',(x,-width/2),(x,width/2))
            for side in [-1,1]:segment(f'{label}_side_{sign}_{side}',(sign*L,side*y),(x,side*y))
        box(f'penalty_mark_{sign}',(sign*(L-f.penalty_mark_distance),0,.0015),(f.mark_size,f.mark_size,.002),white,False)
        p=f.post_width
        # Clear inside width/height. Goal depth measured from field edge to outside rear.
        for side in [-1,1]:
            box(f'post_{sign}_{side}',(sign*(L+p/2),side*(f.goal_width/2+p/2),f.goal_height/2),(p,p,f.goal_height),white)
            box(f'goal_base_{sign}_{side}',(sign*(L+f.goal_depth/2),side*(f.goal_width/2+p/2),p/2),(f.goal_depth,p,p),white)
        box(f'crossbar_{sign}',(sign*(L+p/2),0,f.goal_height+p/2),(p,f.goal_width+2*p,p),white)
        box(f'goal_rear_{sign}',(sign*(L+f.goal_depth-p/2),0,p/2),(p,f.goal_width+2*p,p),white)
    segment('halfway',(0,-W),(0,W))
    box('center_mark',(0,0,.0015),(f.mark_size,f.mark_size,.002),white,False)
    def arc(name,c,r,start,end,n):
        # Outer radius is the specified dimension; segmented visual approximation.
        r-=w/2
        for i in range(n):
            a=start+(end-start)*i/n;b=start+(end-start)*(i+1)/n
            segment(f'{name}_{i}',(c[0]+r*math.cos(a),c[1]+r*math.sin(a)),(c[0]+r*math.cos(b),c[1]+r*math.sin(b)))
    arc('center_circle',(0,0),f.center_circle_diameter/2,0,2*math.pi,96)
    for sx in [-1,1]:
        for sy in [-1,1]:
            start={(1,1):math.pi,(1,-1):math.pi/2,(-1,1):-math.pi/2,(-1,-1):0}[(sx,sy)]
            arc(f'corner_{sx}_{sy}',(sx*L,sy*W),f.corner_radius,start,start+math.pi/2,16)
    return shapes
