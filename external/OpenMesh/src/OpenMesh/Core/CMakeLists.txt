include (VCICommon)

set ( headers
Geometry/Config.hh
Geometry/EigenVectorT.hh
Geometry/LoopSchemeMaskT.hh
Geometry/MathDefs.hh
Geometry/NormalConeT.hh
Geometry/NormalConeT_impl.hh
Geometry/Plane3d.hh
Geometry/QuadricT.hh
Geometry/Vector11T.hh
Geometry/VectorT.hh
Geometry/VectorT_inc.hh
IO/BinaryHelper.hh
IO/IOInstances.hh
IO/IOManager.hh
IO/MeshIO.hh
IO/OFFFormat.hh
IO/OMFormat.hh
IO/OMFormatT_impl.hh
IO/Options.hh
IO/SR_binary.hh
IO/SR_binary_spec.hh
IO/SR_binary_vector_of_bool.hh 
IO/SR_rbo.hh
IO/SR_store.hh
IO/SR_types.hh
IO/StoreRestore.hh
IO/exporter/BaseExporter.hh
IO/exporter/ExporterT.hh
IO/importer/BaseImporter.hh
IO/importer/ImporterT.hh
IO/reader/BaseReader.hh
IO/reader/OBJReader.hh
IO/reader/OFFReader.hh
IO/reader/OMReader.hh
IO/reader/PLYReader.hh
IO/reader/STLReader.hh
IO/writer/BaseWriter.hh
IO/writer/OBJWriter.hh
IO/writer/OFFWriter.hh
IO/writer/OMWriter.hh
IO/writer/PLYWriter.hh
IO/writer/STLWriter.hh
IO/writer/VTKWriter.hh
Mesh/ArrayItems.hh
Mesh/ArrayKernel.hh
Mesh/ArrayKernelT_impl.hh
Mesh/AttribKernelT.hh
Mesh/Attributes.hh
Mesh/BaseKernel.hh
Mesh/BaseMesh.hh
Mesh/Casts.hh
Mesh/CirculatorsT.hh
Mesh/DefaultPolyMesh.hh
Mesh/DefaultTriMesh.hh
Mesh/FinalMeshItemsT.hh
Mesh/Handles.hh
Mesh/IteratorsT.hh
Mesh/PolyConnectivity.hh
Mesh/PolyConnectivity_inline_impl.hh
Mesh/PolyMeshT.hh
Mesh/PolyMeshT_impl.hh
Mesh/PolyMesh_ArrayKernelT.hh
Mesh/SmartHandles.hh
Mesh/SmartRange.hh
Mesh/Status.hh
Mesh/Tags.hh
Mesh/Traits.hh
Mesh/TriConnectivity.hh
Mesh/TriMeshT.hh
Mesh/TriMeshT_impl.hh
Mesh/TriMesh_ArrayKernelT.hh
Mesh/gen/circulators_header.hh
Mesh/gen/circulators_template.hh
Mesh/gen/footer.hh
Mesh/gen/iterators_header.hh
Mesh/gen/iterators_template.hh
System/OpenMeshDLLMacros.hh
System/compiler.hh
System/config.hh
System/mostream.hh
System/omstream.hh
Utils/AutoPropertyHandleT.hh
Utils/BaseProperty.hh
Utils/Endian.hh
Utils/GenProg.hh
Utils/HandleToPropHandle.hh
Utils/Noncopyable.hh
Utils/Predicates.hh
Utils/Property.hh
Utils/PropertyContainer.hh
Utils/PropertyCreator.hh
Utils/PropertyManager.hh
Utils/RandomNumberGenerator.hh
Utils/SingletonT.hh
Utils/SingletonT_impl.hh
Utils/color_cast.hh
Utils/typename.hh
Utils/vector_cast.hh
Utils/vector_traits.hh
)

set ( sources
IO/BinaryHelper.cc
IO/IOManager.cc
IO/OMFormat.cc
IO/reader/BaseReader.cc
IO/reader/OBJReader.cc
IO/reader/OFFReader.cc
IO/reader/OMReader.cc
IO/reader/PLYReader.cc
IO/reader/STLReader.cc
IO/writer/BaseWriter.cc
IO/writer/OBJWriter.cc
IO/writer/OFFWriter.cc
IO/writer/OMWriter.cc
IO/writer/PLYWriter.cc
IO/writer/STLWriter.cc
IO/writer/VTKWriter.cc
Mesh/ArrayKernel.cc
Mesh/BaseKernel.cc
Mesh/PolyConnectivity.cc
Mesh/TriConnectivity.cc
System/omstream.cc
Utils/BaseProperty.cc
Utils/Endian.cc
Utils/PropertyCreator.cc
Utils/RandomNumberGenerator.cc
)

# Disable Library installation when not building OpenMesh on its own but as part of another project!
if ( NOT ${CMAKE_PROJECT_NAME} MATCHES "OpenMesh")
  set(VCI_NO_LIBRARY_INSTALL true)
endif()


if (WIN32)

  if ( OPENMESH_BUILD_SHARED )
    add_definitions( -DOPENMESHDLL -DBUILDOPENMESHDLL)
    vci_add_library (OpenMeshCore SHARED ${sources} ${headers})
  else()
    vci_add_library (OpenMeshCore STATIC ${sources} ${headers})
  endif()

  target_include_directories(OpenMeshCore PUBLIC
                                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                        $<INSTALL_INTERFACE:include>)


else ()
  vci_add_library (OpenMeshCore SHAREDANDSTATIC ${sources} ${headers})

  target_include_directories(OpenMeshCore PUBLIC
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                          $<INSTALL_INTERFACE:include>)

  target_include_directories(OpenMeshCoreStatic PUBLIC
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                          $<INSTALL_INTERFACE:include>)

  set_target_properties (OpenMeshCore PROPERTIES VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
                                               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR} )
endif ()

if (MSVC)
  target_compile_options(OpenMeshCore PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  target_compile_definitions(OpenMeshCore PUBLIC -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif ()

# Add core as dependency before fixbundle 
if ( (${CMAKE_PROJECT_NAME} MATCHES "OpenMesh") AND BUILD_APPS )

  if ( APPLE OR (WIN32 AND NOT "${CMAKE_GENERATOR}" MATCHES "MinGW Makefiles" ) )
      add_dependencies (fixbundle OpenMeshCore)
  endif()

endif()

# if we build debug and release in the same dir, we want to install both!
if ( ${CMAKE_PROJECT_NAME} MATCHES "OpenMesh")
  if ( WIN32 )
    FILE(GLOB files_install_libs "${CMAKE_BINARY_DIR}/Build/lib/*.lib" )
    FILE(GLOB files_install_dlls "${CMAKE_BINARY_DIR}/Build/*.dll" )
    INSTALL(FILES ${files_install_libs} DESTINATION lib )
    INSTALL(FILES ${files_install_dlls} DESTINATION . )
  endif()
endif()


# Install Header Files (Apple)
if ( NOT VCI_PROJECT_MACOS_BUNDLE AND APPLE )
 FILE(GLOB files_install_Geometry    "${CMAKE_CURRENT_SOURCE_DIR}/Geometry/*.hh" )
 FILE(GLOB files_install_IO          "${CMAKE_CURRENT_SOURCE_DIR}/IO/*.hh" "${CMAKE_CURRENT_SOURCE_DIR}/IO/*.inl"  )
 FILE(GLOB files_install_IO_importer "${CMAKE_CURRENT_SOURCE_DIR}/IO/importer/*.hh" )
 FILE(GLOB files_install_IO_exporter "${CMAKE_CURRENT_SOURCE_DIR}/IO/exporter/*.hh" )
 FILE(GLOB files_install_IO_reader   "${CMAKE_CURRENT_SOURCE_DIR}/IO/reader/*.hh" )
 FILE(GLOB files_install_IO_writer   "${CMAKE_CURRENT_SOURCE_DIR}/IO/writer/*.hh" )
 FILE(GLOB files_install_Mesh        "${CMAKE_CURRENT_SOURCE_DIR}/Mesh/*.hh" )
 FILE(GLOB files_install_Mesh_Gen    "${CMAKE_CURRENT_SOURCE_DIR}/Mesh/gen/*.hh" )
 FILE(GLOB files_install_System      "${CMAKE_CURRENT_SOURCE_DIR}/System/*.hh" "${CMAKE_CURRENT_SOURCE_DIR}/System/config.h" )
 FILE(GLOB files_install_Utils       "${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.hh" )
 INSTALL(FILES ${files_install_Geometry}    DESTINATION include/OpenMesh/Core/Geometry )
 INSTALL(FILES ${files_install_IO}          DESTINATION include/OpenMesh/Core/IO )
 INSTALL(FILES ${files_install_IO_importer} DESTINATION include/OpenMesh/Core/IO/importer )
 INSTALL(FILES ${files_install_IO_exporter} DESTINATION include/OpenMesh/Core/IO/exporter )
 INSTALL(FILES ${files_install_IO_reader}   DESTINATION include/OpenMesh/Core/IO/reader )
 INSTALL(FILES ${files_install_IO_writer}   DESTINATION include/OpenMesh/Core/IO/writer )
 INSTALL(FILES ${files_install_Mesh}        DESTINATION include/OpenMesh/Core/Mesh )
 INSTALL(FILES ${files_install_Mesh_Gen}    DESTINATION include/OpenMesh/Core/Mesh/gen )
 INSTALL(FILES ${files_install_System}      DESTINATION include/OpenMesh/Core/System )
 INSTALL(FILES ${files_install_Utils}       DESTINATION include/OpenMesh/Core/Utils )
endif()


# Only install if the project name matches OpenMesh.
if (${CMAKE_PROJECT_NAME} MATCHES "OpenMesh")
  set (OPENMESH_NO_INSTALL_HEADERS FALSE CACHE BOOL "Should OpenMesh skip installing headers?")
else()
  set (OPENMESH_NO_INSTALL_HEADERS TRUE CACHE BOOL "Should OpenMesh skip installing headers?")
endif()

if (NOT APPLE AND NOT ${OPENMESH_NO_INSTALL_HEADERS})

# Install Header Files)
install(DIRECTORY . 
	DESTINATION include/OpenMesh/Core
        FILES_MATCHING 
	PATTERN "*.hh"
        PATTERN "CVS" EXCLUDE
        PATTERN ".svn" EXCLUDE
        PATTERN "tmp" EXCLUDE
	PATTERN "Templates" EXCLUDE
        PATTERN "Debian*" EXCLUDE)

#install the config file
install(FILES System/config.h DESTINATION include/OpenMesh/Core/System)

endif ()

install(TARGETS OpenMeshCore EXPORT OpenMeshConfig
  ARCHIVE DESTINATION ${VCI_PROJECT_LIBDIR}
  LIBRARY DESTINATION ${VCI_PROJECT_LIBDIR}
  RUNTIME DESTINATION ${VCI_PROJECT_BINDIR})

