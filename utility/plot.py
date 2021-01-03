# coding: utf-8
# author: xuxc

__all__ = [
    'contour_plot',
]


def contour_plot(vtk_filename, field_name='Displacement', component=-1,
                 shape_name='Displacement', factor=1):
    """
    绘制云图
    :param vtk_filename: vtk格式的结果文件名
    :param field_name: 需要显示的场变量
    :param component: 场的分量，-1为幅值，0、1、2分别为向量的三个分量
    :param shape_name: 需要显示的变形
    :param factor: 变形缩放系数
    """
    import vtk
    if vtk_filename.endswith('.vtk'):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        print('不能识别的文件格式。')
        return

    line_width = 3
    if component < -1 or component > 2:
        component = -1

    reader.SetFileName(vtk_filename)
    # 默认显示物体变形后的位形。
    # 读取所有的标量场和向量场，以便能够在变形图上显示其它云图
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    # 激活位移向量，以便通过warpVector计算出物体变形后的形态
    reader.SetVectorsName(shape_name)
    reader.Update()

    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(900, 720)
    ren_win.SetWindowName('PyPost')
    ren_win.Render()
    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetRenderWindow(ren_win)
    iren.SetInteractorStyle(style)

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6667, 0.0)  # 让场变量小的地方显示绿色，大的显示红色。默认情况与之相反
    if component == -1:
        lut.SetVectorModeToMagnitude()
    else:
        lut.SetVectorModeToComponent()
        lut.SetVectorComponent(component)

    # 坐标轴
    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(iren)
    axes_widget.SetViewport(0.0, 0.0, 0.15, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()

    # 原始形状
    original_mapper = vtk.vtkDataSetMapper()
    original_mapper.SetInputData(reader.GetOutput())
    original_actor = vtk.vtkActor()
    original_actor.SetMapper(original_mapper)
    original_actor.GetProperty().SetLineWidth(line_width)
    original_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    original_actor.GetProperty().EdgeVisibilityOn()
    ren.AddActor(original_actor)

    # 变形后的形状
    warp = vtk.vtkWarpVector()
    warp.SetInputConnection(reader.GetOutputPort())
    deformation_mapper = vtk.vtkDataSetMapper()
    deformation_mapper.SetInputConnection(warp.GetOutputPort())
    deformation_mapper.SetLookupTable(lut)
    if 'Displacement' in field_name:
        deformation_mapper.SetScalarModeToUsePointFieldData()
        vectors = reader.GetOutput().GetPointData().GetArray(field_name)
    else:
        deformation_mapper.SetScalarModeToUseCellFieldData()
        vectors = reader.GetOutput().GetCellData().GetArray(field_name)
    # 在变形后的形态上显示选择的场变量
    deformation_mapper.SelectColorArray(field_name)
    deformation_mapper.SetScalarRange(vectors.GetRange(component))
    deformation_actor = vtk.vtkActor()
    deformation_actor.SetMapper(deformation_mapper)
    deformation_actor.GetProperty().SetLineWidth(line_width+2)
    warp.SetScaleFactor(factor)
    ren.AddActor(deformation_actor)

    # 图例
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(deformation_mapper.GetLookupTable())
    if component == 0:
        text = field_name + ' X'
    elif component == 1:
        text = field_name + ' Y'
    elif component == 2:
        text = field_name + ' Z'
    else:
        text = field_name
    scalar_bar.SetTitle('{}\n'.format(text))
    scalar_bar.SetNumberOfLabels(8)
    scalar_bar.SetPosition(0.05, 0.2)
    scalar_bar.SetPosition2(0.1, 0.75)
    scalar_bar.SetLabelFormat("%5.3e")
    prop_title = vtk.vtkTextProperty()
    prop_label = vtk.vtkTextProperty()
    prop_title.SetFontFamilyToArial()
    prop_title.ItalicOff()
    prop_title.BoldOn()
    prop_title.SetColor(0.1, 0.1, 0.1)
    prop_label.BoldOff()
    prop_label.SetColor(0.1, 0.1, 0.1)
    scalar_bar.SetTitleTextProperty(prop_title)
    scalar_bar.SetLabelTextProperty(prop_label)
    ren.AddActor(scalar_bar)

    ren.SetBackground(0.9, 0.9, 0.9)
    ren.ResetCamera()

    iren.Initialize()
    iren.Start()
