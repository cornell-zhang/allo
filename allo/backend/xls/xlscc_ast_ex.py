from xlscc_nodes import *

def build_example_testblock_ast():
    inc = XLSCCInclude("cstdint", angled=True)

    tmpl_params = [
        XLSCCTemplateParam("int", "Width"),
        XLSCCTemplateParam("bool", "Signed", default=XLSCCLiteral(True)),
    ]
    aliased_type = XLSCCType(
        "XlsInt",
        template_args=[XLSCCVar("Width"), XLSCCVar("Signed")],
    )
    ac_int_alias = XLSCCTemplateAlias(
        template_params=tmpl_params,
        alias_name="ac_int",
        aliased_type=aliased_type,
    )

    ac_int32 = XLSCCType(
        "ac_int",
        template_args=[XLSCCLiteral(32), XLSCCLiteral(True)],
    )

    add_params = [
        XLSCCParam(ac_int32, "a"),
        XLSCCParam(ac_int32, "b"),
    ]
    add_body = [
        XLSCCReturnStmt(
            XLSCCBinOp("+", XLSCCVar("a"), XLSCCVar("b"))
        )
    ]
    add_method = XLSCCMethod(
        name="add",
        return_type=ac_int32,
        params=add_params,
        body=add_body,
    )

    int32_type = XLSCCType("int32_t")
    int_type = XLSCCType("int")

    run_params = [
        XLSCCParam(int32_type, "a"),
        XLSCCParam(int32_type, "b"),
        XLSCCParam(int_type, "use"),
    ]
    run_body = [
        XLSCCReturnStmt(
            XLSCCCall("add", [XLSCCVar("a"), XLSCCVar("b")])
        )
    ]
    run_method = XLSCCMethod(
        name="Run",
        return_type=int32_type,
        params=run_params,
        body=run_body,
        pragmas=[XLSCCPragma("hls_top")],
    )

    testblock_class = XLSCCClass(
        name="TestBlock",
        members=[
            XLSCCAccessSpec("private"),
            add_method,
            XLSCCAccessSpec("public"),
            run_method,
        ],
    )

    tu = XLSCCTranslationUnit(
        includes=[inc],
        decls=[ac_int_alias, testblock_class],
    )
    return tu
