import pandas as pd
from sqlalchemy import select, cast, func
from typing import Literal

from database.engine import engine
from database.tables import *

def fetchGWMeanVarStd(
        GWTPoint: int
        ) -> tuple[float]:
    """Fetch the multi-annual mean depth of the groundwater table for the provided GWT point
    and its variance and standard deviation.

    Args:
        GWTPoint (int): ID of the GWT point

    Returns:
        tuple[float]: mean, variance and standar deviation of groundwater table depth (m)
    """    

    stmt = (
        select(
            GWTPoints.c['GWDepth', 'GWDepthStd', 'GWDepthVar']
        ).where(
            GWTPoints.c['ID'] == GWTPoint
        )
    )

    with engine.connect() as conn:
        data = conn.execute(stmt).fetchone()

    return data


def fetchGST(
        GWTPoint: int,
        UHI_partition: Literal['AreaType', 'SurfImper']
        ) -> pd.DataFrame:
    """Fetch the GST for the respective UHI unit where the given GWT Point is locacted in

    Args:
        GWTPoint (int): ID of the GWT point
        UHI_partition (Literal[AreaType, SurfImper]): GST according to Area Type  or Surface Imperviousness partiton

    Raises:
        KeyError: In case the provided GWT point lays in a UHI unit that has no records. Partiton by Area Type has
        full coverage, while partiton by Surface Imperviousness has not.

    Returns:
        pd.DataFrame
    """    
    
    min_year = (
        select(
            func.min(GWTRec.c['Year'])
        ).where(
            GWTRec.c['GWTPoint'] == GWTPoint
        )
    )

    min_month = (
        select(
            func.min(GWTRec.c['Month'])
        ).where(
            GWTRec.c['GWTPoint'] == GWTPoint,
            GWTRec.c['Year'] == min_year.scalar_subquery()
        )
    )
    
    stmt = (
        select(
            (cast(GSTRec.c['Year'], String) + "-" + cast(GSTRec.c['Month'], String)).label('YearMonth'),
            GSTRec.c['Year'].label('Year'),
            GSTRec.c['Month'].label('Month'),
            GSTRec.c['T'].label('GST')
        ).select_from(
            GSTRec.join(
                UHIUnits, GSTRec.c['UHIUnit'] == UHIUnits.c['ID']
            ).join(
                GWTPoints, UHIUnits.c['ID'] == GWTPoints.c[UHI_partition + 'ID']
            )
        ).where(
            GWTPoints.c['ID'] == GWTPoint,
            GSTRec.c['Year'] >= min_year.scalar_subquery()
        )
    )
    
    with engine.connect() as conn:
        data = conn.execute(stmt).fetchall()
        min_year = conn.execute(min_year).fetchone()[0]
        min_month = conn.execute(min_month).fetchone()[0]
    
    if not data:
        raise KeyError(f'The provided GWT point is located in a {UHI_partition} unit that has no existing GST records')

    df = pd.DataFrame(data)

    #filter out the Year-Month that came beofre IC
    ic_idx = df[(df['Year'] == min_year) & (df['Month'] == min_month)].index[0] #get the index of the IC Year-Month
    df = df.iloc[ic_idx:].reset_index(drop=True) #start from there and reset index to 0
    df['t'] = df.index*(365.25*24*60*60/12) #avg amount of sec in a calendar month --- solver times

    return df
    
def fetchDarcyFlux(
        GWTPoint: int,
        K: Literal['MIN', 'MAX'] = 'MIN'
        ) -> float:
    """Fetch the Darcy flux rate for the given GWT point

    Args:
        GWTPoint (int): ID of the GWT point
        K (Literal[MIN, MAX]): Choice of the hydraulic conductivity - minimal, maximal estimates. Defaults to 'MIN'

    Returns:
        float
    """    
    
    Keq_stmt = (
        select(
            func.sum(GWTGeo.c['Z1'] - GWTGeo.c['Z0']), #total height of soil column - numerator
            func.sum((GWTGeo.c['Z1'] - GWTGeo.c['Z0'])/Soils.c['K'+K]) #thickness averaged K - denominator
        ).select_from(
            GWTGeo.join(
                Soils, GWTGeo.c['SoilID'] == Soils.c['ID']
            )
        ).where(
            GWTGeo.c['GWTPoint'] == GWTPoint
        )
    )

    grad_stmt = (
        select(
            TopoUnits.c['HydroGrad']
        ).select_from(
            TopoUnits
            .join(
                GWTPoints, TopoUnits.c['ID'] == GWTPoints.c['TopoUnitID']
            )
        ).where(
            GWTPoints.c['ID'] == GWTPoint
        )
    )
    
    with engine.connect() as conn:
        num, denom = conn.execute(Keq_stmt).fetchall()[0]
        grad = conn.execute(grad_stmt).fetchone()[0]

    return (num/denom)*grad


def fetchMeasGWT(
        GWTPoint: int
        ) -> pd.DataFrame:
    """Fetch the measured groundwater temperatures for the given GWT point

    Args:
        GWTPoint (int): ID of the GWT point

    Returns:
        pd.DataFrame
    """    


    stmt = (
        select(
            (cast(GWTRec.c['Year'], String) + "-" + cast(GWTRec.c['Month'], String)).label('YearMonth'),
            GWTRec.c['Year'].label('Year'),
            GWTRec.c['Month'].label('Month'),
            GWTRec.c['Z'].label('Z'),
            func.avg(GWTRec.c['T']).label('T'),
        ).group_by(
            GWTRec.c['Year', 'Month', 'Day', 'Z']
        ).order_by(
            GWTRec.c['Year', 'Month', 'Day', 'Z']
        ).where(
            GWTRec.c['GWTPoint'] == GWTPoint
        )
    )

    with engine.connect() as conn:
        data = conn.execute(stmt).fetchall()

    return pd.DataFrame(data)

def fetchGeoProfile(
        GWTPoint: int,
        lamb: Literal['MIN', 'MAX', 'REC'] = 'REC',
        C: Literal['MIN', 'MAX'] = 'MAX'
        ) -> pd.DataFrame:
    """Fetch the vertical geological profile for the provided GWT point

    Args:
        GWTPoint (int): ID of the GWT point
        lamb (Literal[MIN, MAX, REC]): Choice of the thermal conductivity - minimal, maximal or recommended estimates. Defaults to 'REC'
        C (Literal[MIN, MAX]): Choice of the bulk volumetric heat capacity - minimal, maximal estimates (unsaturated and saturated). Defaults to 'MAX'

    Returns:
        pd.DataFrame: start and end depth of each layer with the corresponding thermal conductivity, bulk volumetric heat capacity and soil type
    """    
    
    max_depth = (
        select(
            func.max(GWTRec.c['Z'])
        ).where(
            GWTRec.c['GWTPoint'] == GWTPoint
        )
    )

    stmt = (
        select(
            GWTGeo.c['Z0'].label('Z0'),
            GWTGeo.c['Z1'].label('Z1'),
            Soils.c['Lamb' + lamb].label('Lamb'),
            Soils.c['C' + C].label('C'),
            Soils.c['Name'].label('Soil')
        ).select_from(
            GWTGeo.join(
                Soils, GWTGeo.c['SoilID'] == Soils.c['ID']
            )
        ).where(
            GWTGeo.c['GWTPoint'] == GWTPoint,
            GWTGeo.c['Z0'] <= max_depth.scalar_subquery() #trim the geolgical profile at the bottom most GWT measurment
        )
    )

    with engine.connect() as conn:
        data = conn.execute(stmt).fetchall()
        max_depth = conn.execute(max_depth).fetchone()[0]

    df = pd.DataFrame(data)
    df.loc[df.index[-1], 'Z1'] = max_depth #set the depth to the bottom most GWT measurment

    return df
