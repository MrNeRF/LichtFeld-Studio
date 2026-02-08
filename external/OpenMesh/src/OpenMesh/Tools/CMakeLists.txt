include (VCICommon)

set ( headers  
Decimater/BaseDecimaterT.hh
Decimater/BaseDecimaterT_impl.hh
Decimater/CollapseInfoT.hh
Decimater/DecimaterT.hh
Decimater/DecimaterT_impl.hh
Decimater/McDecimaterT.hh
Decimater/McDecimaterT_impl.hh
Decimater/MixedDecimaterT.hh
Decimater/MixedDecimaterT_impl.hh
Decimater/ModAspectRatioT.hh
Decimater/ModAspectRatioT_impl.hh
Decimater/ModBaseT.hh
Decimater/ModEdgeLengthT.hh
Decimater/ModEdgeLengthT_impl.hh
Decimater/ModHausdorffT.hh
Decimater/ModHausdorffT_impl.hh
Decimater/ModIndependentSetsT.hh
Decimater/ModNormalDeviationT.hh
Decimater/ModNormalFlippingT.hh
Decimater/ModProgMeshT.hh
Decimater/ModProgMeshT_impl.hh
Decimater/ModQuadricT.hh
Decimater/ModQuadricT_impl.hh
Decimater/ModRoundnessT.hh
Decimater/Observer.hh
Dualizer/meshDualT.hh
HoleFiller/HoleFillerT.hh
HoleFiller/HoleFillerT_impl.hh
SmartTagger/SmartTaggerT.hh
SmartTagger/SmartTaggerT_impl.hh
Smoother/JacobiLaplaceSmootherT.hh
Smoother/JacobiLaplaceSmootherT_impl.hh
Smoother/LaplaceSmootherT.hh
Smoother/LaplaceSmootherT_impl.hh
Smoother/SmootherT.hh
Smoother/SmootherT_impl.hh
Smoother/smooth_mesh.hh
Subdivider/Adaptive/Composite/CompositeT.hh
Subdivider/Adaptive/Composite/CompositeT_impl.hh
Subdivider/Adaptive/Composite/CompositeTraits.hh
Subdivider/Adaptive/Composite/RuleInterfaceT.hh
Subdivider/Adaptive/Composite/RulesT.hh
Subdivider/Adaptive/Composite/RulesT_impl.hh
Subdivider/Adaptive/Composite/Traits.hh
Subdivider/Uniform/Composite/CompositeT.hh
Subdivider/Uniform/Composite/CompositeT_impl.hh
Subdivider/Uniform/Composite/CompositeTraits.hh
Subdivider/Uniform/CatmullClarkT.hh
Subdivider/Uniform/CatmullClarkT_impl.hh
Subdivider/Uniform/CompositeLoopT.hh
Subdivider/Uniform/CompositeSqrt3T.hh
Subdivider/Uniform/LongestEdgeT.hh
Subdivider/Uniform/LoopT.hh
Subdivider/Uniform/MidpointT.hh
Subdivider/Uniform/ModifiedButterFlyT.hh
Subdivider/Uniform/Sqrt3InterpolatingSubdividerLabsikGreinerT.hh
Subdivider/Uniform/Sqrt3T.hh
Subdivider/Uniform/SubdividerT.hh
Utils/Config.hh
Utils/GLConstAsString.hh
Utils/Gnuplot.hh
Utils/HeapT.hh
Utils/MeshCheckerT.hh
Utils/MeshCheckerT_impl.hh
Utils/NumLimitsT.hh
Utils/StripifierT.hh
Utils/StripifierT_impl.hh
Utils/TestingFramework.hh
Utils/Timer.hh
Utils/conio.hh
VDPM/MeshTraits.hh
VDPM/StreamingDef.hh
VDPM/VFront.hh
VDPM/VHierarchy.hh
VDPM/VHierarchyNode.hh
VDPM/VHierarchyNodeIndex.hh
VDPM/VHierarchyWindow.hh
VDPM/ViewingParameters.hh
)


set ( sources
Decimater/Observer.cc
Utils/Gnuplot.cc
Utils/Timer.cc
Utils/conio.cc
VDPM/VFront.cc
VDPM/VHierarchy.cc
VDPM/VHierarchyNodeIndex.cc
VDPM/VHierarchyWindow.cc
VDPM/ViewingParameters.cc
)


IF(WIN32 AND NOT MINGW)
  list(APPEND sources "Utils/getopt.c")
ENDIF(WIN32 AND NOT MINGW)

# Disable Library installation when not building OpenMesh on its own but as part of another project!
if ( NOT ${CMAKE_PROJECT_NAME} MATCHES "OpenMesh")
  set(VCI_NO_LIBRARY_INSTALL true)
endif()


if (WIN32)

  if ( OPENMESH_BUILD_SHARED )
    add_definitions( -DOPENMESHDLL -DBUILDOPENMESHDLL)
    vci_add_library (OpenMeshTools SHARED ${sources} ${headers})
  else()
    # OpenMesh has no dll exports so we have to build a static library on windows
    vci_add_library (OpenMeshTools STATIC ${sources} ${headers})
  endif()

  target_include_directories(OpenMeshTools PUBLIC
                                        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                        $<INSTALL_INTERFACE:include>)

else ()
  vci_add_library (OpenMeshTools SHAREDANDSTATIC ${sources} ${headers})

  target_include_directories(OpenMeshTools PUBLIC
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                          $<INSTALL_INTERFACE:include>)

  target_include_directories(OpenMeshToolsStatic PUBLIC
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
                                          $<INSTALL_INTERFACE:include>)

  set_target_properties (OpenMeshTools PROPERTIES VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
                                                SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR} )
endif ()

target_link_libraries (OpenMeshTools OpenMeshCore)

IF( NOT WIN32 )
  target_link_libraries (OpenMeshToolsStatic OpenMeshCoreStatic)
ENDIF(NOT WIN32)

if ( (${CMAKE_PROJECT_NAME} MATCHES "OpenMesh") AND BUILD_APPS )

  if ( WIN32 )
    if ( NOT "${CMAKE_GENERATOR}" MATCHES "MinGW Makefiles" )
      # let bundle generation depend on target
      add_dependencies (fixbundle OpenMeshTools)
    endif()
  endif()

  # Add tools as dependency before fixbundle 
  if (APPLE)
    # let bundle generation depend on target
    add_dependencies (fixbundle OpenMeshTools)
    add_dependencies (fixbundle OpenMeshToolsStatic)
  endif()

endif()


# Install Header Files (Apple)
if ( NOT VCI_PROJECT_MACOS_BUNDLE AND APPLE )
 FILE(GLOB files_install_Decimater                       "${CMAKE_CURRENT_SOURCE_DIR}/Decimater/*.hh" )
 FILE(GLOB files_install_Dualizer                        "${CMAKE_CURRENT_SOURCE_DIR}/Dualizer/*.hh" )
 FILE(GLOB files_install_HoleFiller                      "${CMAKE_CURRENT_SOURCE_DIR}/HoleFiller/*.hh" )
 FILE(GLOB files_install_KERNEL_OSG                      "${CMAKE_CURRENT_SOURCE_DIR}/Kernel_OSG/*.hh" )
 FILE(GLOB files_install_Smoother                        "${CMAKE_CURRENT_SOURCE_DIR}/Smoother/*.hh" )
 FILE(GLOB files_install_SmartTagger                     "${CMAKE_CURRENT_SOURCE_DIR}/SmartTagger/*.hh" )
 FILE(GLOB files_install_Subdivider_Adaptive             "${CMAKE_CURRENT_SOURCE_DIR}/Subdivider/Adaptive/Composite/*.hh" )
 FILE(GLOB files_install_Subdivider_Uniform              "${CMAKE_CURRENT_SOURCE_DIR}/Subdivider/Uniform/*.hh" )
 FILE(GLOB files_install_Subdivider_Uniform_Composite    "${CMAKE_CURRENT_SOURCE_DIR}/Subdivider/Uniform/Composite/*.hh" )
 FILE(GLOB files_install_Utils                           "${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.hh" "${CMAKE_CURRENT_SOURCE_DIR}/Utils/getopt.h" )
 FILE(GLOB files_install_VDPM                            "${CMAKE_CURRENT_SOURCE_DIR}/VDPM/*.hh" )
 INSTALL(FILES ${files_install_Decimater}                       DESTINATION include/OpenMesh/Tools/Decimater )
 INSTALL(FILES ${files_install_Dualizer}                        DESTINATION include/OpenMesh/Tools/Dualizer )
 INSTALL(FILES ${files_install_HoleFiller}                      DESTINATION include/OpenMesh/Tools/HoleFiller )
 INSTALL(FILES ${files_install_KERNEL_OSG}                      DESTINATION include/OpenMesh/Tools/Kernel_OSG )
 INSTALL(FILES ${files_install_Smoother}                        DESTINATION include/OpenMesh/Tools/Smoother )
 INSTALL(FILES ${files_install_SmartTagger}                     DESTINATION include/OpenMesh/Tools/SmartTagger )
 INSTALL(FILES ${files_install_Subdivider_Adaptive}             DESTINATION include/OpenMesh/Tools/Subdivider/Adaptive/Composite )
 INSTALL(FILES ${files_install_Subdivider_Uniform}              DESTINATION include/OpenMesh/Tools/Subdivider/Uniform )
 INSTALL(FILES ${files_install_Subdivider_Uniform_Composite}    DESTINATION include/OpenMesh/Tools/Subdivider/Uniform/Composite )
 INSTALL(FILES ${files_install_Utils}                           DESTINATION include/OpenMesh/Tools/Utils )
 INSTALL(FILES ${files_install_VDPM}                            DESTINATION include/OpenMesh/Tools/VDPM )
endif()


# Only install if the project name matches OpenMesh.
if (${CMAKE_PROJECT_NAME} MATCHES "OpenMesh")
  set (OPENMESH_NO_INSTALL_HEADERS FALSE CACHE BOOL "Should OpenMesh skip installing headers?")
else()
  set (OPENMESH_NO_INSTALL_HEADERS TRUE CACHE BOOL "Should OpenMesh skip installing headers?")
endif()

if (NOT APPLE AND NOT ${OPENMESH_NO_INSTALL_HEADERS})
  # Install Header Files
  install(DIRECTORY .
        DESTINATION include/OpenMesh/Tools
        FILES_MATCHING
        PATTERN "*.hh"
        PATTERN "CVS" EXCLUDE
        PATTERN ".svn" EXCLUDE
        PATTERN "tmp" EXCLUDE
        PATTERN "Templates" EXCLUDE
        PATTERN "Debian*" EXCLUDE)

  #install the config file
  install(FILES Utils/getopt.h DESTINATION include/OpenMesh/Tools/Utils)
endif ()

install(TARGETS OpenMeshTools EXPORT OpenMeshConfig
  ARCHIVE DESTINATION ${VCI_PROJECT_LIBDIR}
  LIBRARY DESTINATION ${VCI_PROJECT_LIBDIR}
  RUNTIME DESTINATION ${VCI_PROJECT_BINDIR})

