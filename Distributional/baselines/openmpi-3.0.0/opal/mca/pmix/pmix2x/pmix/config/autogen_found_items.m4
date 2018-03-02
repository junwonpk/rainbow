dnl
dnl $HEADER$
dnl
dnl ---------------------------------------------------------------------------
dnl This file is automatically created by autogen.pl; it should not
dnl be edited by hand!!
dnl
dnl Generated by ec2-user at Tue Sep 12 21:31:45 2017
dnl on ip-172-31-1-185.
dnl ---------------------------------------------------------------------------

m4_define([autogen_platform_file], [])


dnl Project names
m4_define([project_name_long], [PMIx])
m4_define([project_name_short], [PMIx])

dnl ---------------------------------------------------------------------------
dnl ---------------------------------------------------------------------------
dnl ---------------------------------------------------------------------------

dnl MCA information
dnl ---------------------------------------------------------------------------

dnl Frameworks in the pmix project and their corresponding directories
m4_define([mca_pmix_framework_list], [pdl, pif, pinstalldirs, pnet, psec, psensor, ptl])

dnl Components in the pmix / pdl framework
m4_define([mca_pmix_pdl_m4_config_component_list], [pdlopen, plibltdl])
m4_define([mca_pmix_pdl_no_config_component_list], [])

dnl Components in the pmix / pif framework
m4_define([mca_pmix_pif_m4_config_component_list], [bsdx_ipv4, bsdx_ipv6, linux_ipv6, posix_ipv4, solaris_ipv6])
m4_define([mca_pmix_pif_no_config_component_list], [])

dnl Components in the pmix / pinstalldirs framework
m4_define([mca_pmix_pinstalldirs_m4_config_component_list], [config, env])
m4_define([mca_pmix_pinstalldirs_no_config_component_list], [])

dnl Components in the pmix / pnet framework
m4_define([mca_pmix_pnet_m4_config_component_list], [opa])
m4_define([mca_pmix_pnet_no_config_component_list], [])

dnl Components in the pmix / psec framework
m4_define([mca_pmix_psec_m4_config_component_list], [munge])
m4_define([mca_pmix_psec_no_config_component_list], [native, none])

dnl Components in the pmix / psensor framework
m4_define([mca_pmix_psensor_m4_config_component_list], [])
m4_define([mca_pmix_psensor_no_config_component_list], [file, heartbeat])

dnl Components in the pmix / ptl framework
m4_define([mca_pmix_ptl_m4_config_component_list], [])
m4_define([mca_pmix_ptl_no_config_component_list], [tcp, usock])

dnl ---------------------------------------------------------------------------

dnl List of configure.m4 files to include
m4_include([src/mca/pdl/configure.m4])
m4_include([src/mca/pinstalldirs/configure.m4])
m4_include([src/mca/pdl/pdlopen/configure.m4])
m4_include([src/mca/pdl/plibltdl/configure.m4])
m4_include([src/mca/pif/bsdx_ipv4/configure.m4])
m4_include([src/mca/pif/bsdx_ipv6/configure.m4])
m4_include([src/mca/pif/linux_ipv6/configure.m4])
m4_include([src/mca/pif/posix_ipv4/configure.m4])
m4_include([src/mca/pif/solaris_ipv6/configure.m4])
m4_include([src/mca/pinstalldirs/config/configure.m4])
m4_include([src/mca/pinstalldirs/env/configure.m4])
m4_include([src/mca/pnet/opa/configure.m4])
m4_include([src/mca/psec/munge/configure.m4])
