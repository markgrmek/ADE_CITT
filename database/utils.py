from sqlalchemy import func

def sqlmean(column):
    return func.avg(column)

def sqlstd(column):
    return func.sqrt(func.avg(column * column) - func.pow(func.avg(column), 2))

def sqlvar(column):
    return func.avg(column * column) - func.pow(func.avg(column), 2)
