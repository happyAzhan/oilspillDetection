import turtle as t
import pandas as pd
data = pd.read_excel("./data/4609_.xlsx").values


t.color('black','black')

t.screensize(canvwidth=696, canvheight=682, bg='white')
t.hideturtle()
# t.tracer(0)
# t.penup()
# t.goto(260-341,-150+348)
# t.pendown()
# t.dot(2)
# t.update()
# t.tracer(0)
# t.penup()
# t.goto(560-341,-450+348)
# t.pendown()
# t.dot(2)
# t.update()
for row in data:
    t.tracer(0)
    t.penup()
    t.goto(row[0]-341,-row[1]+348)
    t.pendown()
    t.dot(2)
    t.update()
t.done()
