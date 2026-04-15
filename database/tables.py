from sqlalchemy import Table, Column, String, Integer, Float, ForeignKey, UniqueConstraint
from database.engine import metadata

#===================================================================================
#PRIMARY TABLES
#===================================================================================
UHIUnits = Table(
    'UHIUnits',
    metadata,
    Column('ID', Integer, primary_key=True),
    Column('Name', String)
)

TopoUnits = Table(
    'TopoUnits',
    metadata,
    Column('ID', Integer, primary_key=True),
    Column('Name', String, unique = True),
    Column('HydroGrad', Float),
    Column('HydroGradStd', Float),
    Column('HydroGradVar', Float)
)


Soils = Table(
    'Soils',
    metadata,
    Column('ID', Integer, primary_key=True, autoincrement=True),
    Column('GeoUnit', String),
    Column('Name', String, unique=True),
    Column('LambMIN', Float),
    Column('LambMAX', Float),
    Column('LambREC', Float),
    Column('CMIN', Float),
    Column('CMAX', Float),
    Column('KMIN', Float),
    Column('KMAX', Float)
)

#===================================================================================
#SECONDARY TABLES
#===================================================================================

GWTPoints = Table(
    'GWTPoints',
    metadata,
    Column('ID', Integer, primary_key=True),
    Column('UTME', Float),
    Column('UTMN', Float),
    Column('Depth', Float),
    Column('GWDepth', Float),
    Column('GWDepthStd', Float),
    Column('GWDepthVar', Float),
    Column('TopoUnitID', String, ForeignKey('TopoUnits.ID')),
    Column('AreaTypeID', Integer, ForeignKey('UHIUnits.ID')),
    Column('SurfImperID', Integer, ForeignKey('UHIUnits.ID')),
    Column('LocDesc', String)
    )

#===================================================================================
#DATA TABLES
#===================================================================================

GWTRec = Table(
    'GWTRec',
    metadata,
    Column('GWTPoint', Integer, ForeignKey('GWTPoints.ID')),
    Column('Year', Integer),
    Column('Month', Integer),
    Column('Day', Integer),
    Column('Z', Float),
    Column('T', Float),
    UniqueConstraint('GWTPoint', 'Year', 'Month', 'Day', 'Z')
    )

GWTGeo = Table(
    'GWTGeo',
    metadata,
    Column('GWTPoint', Integer, ForeignKey('GWTPoints.ID')),
    Column('Z0', Float),
    Column('Z1', Float),
    Column('SoilID', Integer, ForeignKey('Soils.ID')),
    UniqueConstraint('GWTPoint', 'Z0', 'Z1')
)

GSTRec = Table(
    'GSTRec',
    metadata,
    Column('UHIUnit', Integer, ForeignKey('UHIUnits.ID')),
    Column('Year', Integer),
    Column('Month', Integer),
    Column('T', Float),
    UniqueConstraint('UHIUnit', 'Year', 'Month')
)



